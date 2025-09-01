## 1. Setup and Dependencies
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import csv
import os
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List, Union, Dict
import pandas as pd
import time
import re
from openai import RateLimitError
from dotenv import load_dotenv
import httpx
import logging

# Set up logging
logging.basicConfig(filename='agent_debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## 2. Data Loading
def load_json_file(file_path, default_value=None, required=False):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully loaded {file_path}")
            return data
    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found.")
        print(f"Error: {file_path} not found.")
        if required:
            print(f"Execution halted: {file_path} is required. Please upload it to /Data/.")
            exit(1)
        return default_value or {}
    except json.JSONDecodeError as e:
        logging.error(f"Error: {file_path} is invalid JSON. Details: {e}")
        print(f"Error: {file_path} is invalid JSON. Details: {e}")
        if required:
            exit(1)
        return default_value or {}

reference_codes = load_json_file('./Data/reference_codes.json', {'ICD-10': {}, 'CPT': {}}, required=True)
print(f"Loaded reference_codes: {list(reference_codes.keys())}")
insurance_policies = load_json_file('./Data/insurance_policies.json', [], required=True)
insurance_policies_map = {policy['policy_id']: policy for policy in insurance_policies}
print(f"Loaded insurance_policies: {len(insurance_policies)} policies")
validation_records = load_json_file('./Data/validation_records.json', [], required=True)
print(f"Loaded validation_records: {len(validation_records)} records")
test_records = load_json_file('./Data/test_records.json', [], required=True)
print(f"Loaded test_records: {len(test_records)} records")

def load_data(file_path, required=False):
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        return {str(row['patient_id']): row.to_dict() for _, row in df.iterrows()}
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        if required:
            print(f"Execution halted: {file_path} is required. Please upload it to /Data/.")
            exit(1)
        return {}
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        print(f"Error loading {file_path}: {e}")
        if required:
            exit(1)
        return {}

validation_reference_results = load_data('./Data/validation_reference_results.xlsx', required=True)


## 3. Data Preprocessing
def compute_age(dob, dos):
    try:
        dob_date = date.fromisoformat(dob)
        dos_date = date.fromisoformat(dos)
        age = dos_date.year - dob_date.year
        if (dos_date.month, dos_date.day) < (dob_date.month, dob_date.day):
            age -= 1
        return age
    except ValueError as e:
        logging.warning(f"Invalid date format: dob={dob}, dos={dos}. Error: {e}")
        print(f"Warning: Invalid date format: dob={dob}, dos={dos}. Using default age 0.")
        return 0

def preprocess_records(records):
    for record in records:
        if 'date_of_birth' in record and 'date_of_service' in record:
            age = compute_age(record['date_of_birth'], record['date_of_service'])
            record['age'] = age if age is not None else 'Unknown'
        else:
            record['age'] = 'Unknown'
    return records

validation_records = preprocess_records(validation_records)
test_records = preprocess_records(test_records)


## 4. Azure OpenAI Configuration
def get_access_token():
    auth = "https://api.uhg.com/oauth2/token"
    scope = "https://api.uhg.com/.default"
    grant_type = "client_credentials"
    try:
        client_id = dbutils.secrets.get(scope="AIML_Training", key="client_id")
        client_secret = dbutils.secrets.get(scope="AIML_Training", key="client_secret")
        if not client_id or not client_secret:
            raise ValueError("Client ID or Client Secret not found in Colab secrets. Please add them under Secrets in Colab settings.")
        with httpx.Client() as client:
            body = {
                "grant_type": grant_type,
                "scope": scope,
                "client_id": client_id,
                "client_secret": client_secret,
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = client.post(auth, headers=headers, data=body, timeout=60)
            resp.raise_for_status()
            access_token = resp.json()["access_token"]
            logging.info("Successfully obtained access token.")
            return access_token
    except Exception as e:
        logging.error(f"Error obtaining access token: {e}")
        print(f"Error obtaining access token: {e}")
        return None

# Load environment variables from .env file
load_dotenv('./Data/UAIS_vars.env')

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
EMBEDDINGS_DEPLOYMENT_NAME = os.environ["EMBEDDINGS_DEPLOYMENT_NAME"]
MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]
PROJECT_ID = os.environ['PROJECT_ID']

# Ensure token is valid before initializing chat_client
access_token = get_access_token()
if not access_token:
    print("Execution halted: Failed to obtain a valid access token. Please add client_id and client_secret as Colab secrets under Settings > Secrets.")
    exit(1)

chat_client = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    azure_deployment=MODEL_DEPLOYMENT_NAME,
    temperature=0,
    azure_ad_token=access_token,
    default_headers={"projectId": PROJECT_ID}
)


## 5. Tool Definitions
@tool
def summarize_patient_record(record_str: str) -> str:
    """Extracts a structured summary from a patient record using ICD-10 and CPT mappings."""
    try:
        record = json.loads(record_str)
        if not all(key in record for key in ['patient_id', 'date_of_birth', 'date_of_service']):
            return "Error: Missing required fields in patient record."
    except json.JSONDecodeError:
        return "Error: Invalid patient record format."
 
    prompt = ChatPromptTemplate.from_template(
        "You are a medical claims summarization assistant. Extract a structured summary from the patient record. Do not evaluate claims. Use ICD-10 and CPT mappings.\n\nPatient Record: {record_str}\nICD-10 Mappings: {icd10_mappings}\nCPT Mappings: {cpt_mappings}\n\nOutput:\n- Patient Demographics: [Gender, Age]\n- Insurance Policy ID: [ID]\n- Diagnoses and Descriptions: [ICD-10 codes with descriptions]\n- Procedures and Descriptions: [CPT codes with descriptions]\n- Preauthorization Status: [Obtained/Not obtained, Required: Yes/No]\n- Billed Amount: [USD]\n- Date of Service: [Date]"
    )
    response = chat_client.invoke(prompt.format(
        record_str=record_str,
        icd10_mappings=json.dumps(reference_codes.get('ICD-10', {})),
        cpt_mappings=json.dumps(reference_codes.get('CPT', {}))
    ))
    logging.info(f"Patient Summary for {record.get('patient_id')}: {response.content}")
    return response.content
 
 
@tool
def summarize_policy_guideline(policy_id: str) -> str:
    """Extracts a structured summary from an insurance policy using ICD-10 and CPT mappings."""
    if policy_id not in insurance_policies_map:
        return f"Error: Policy ID {policy_id} not found in the policy database."
 
    policy = insurance_policies_map.get(policy_id)
    prompt = ChatPromptTemplate.from_template(
        "You are an insurance policy summarization assistant. Extract a structured summary from the policy. Do not evaluate claims. Use ICD-10 and CPT mappings.\n\nPolicy: {policy}\nICD-10 Mappings: {icd10_mappings}\nCPT Mappings: {cpt_mappings}\n\nOutput:\n- Policy Details: [Policy ID]\n- Covered Procedures: [List each with sub-points]\n  - Procedure: [CPT code: Description]\n  - Covered Diagnoses: [ICD-10 codes]\n  - Age Range: [Min-Max]\n  - Gender: [Requirement]\n  - Preauthorization Required: [Yes/No]"
    )
    response = chat_client.invoke(prompt.format(
        policy=json.dumps(policy),
        icd10_mappings=json.dumps(reference_codes.get('ICD-10', {})),
        cpt_mappings=json.dumps(reference_codes.get('CPT', {}))
    ))
    logging.info(f"Policy Summary for {policy_id}: {response.content}")
    return response.content
 
 
@tool
def check_claim_coverage(record_summary: str, policy_summary: str) -> str:
    """Validates a claim against policy coverage criteria including diagnosis, procedure, age, gender, and preauthorization."""
    prompt = ChatPromptTemplate.from_template(
        "You are a claims validation specialist. Determine if the claim meets coverage criteria: diagnosis match, procedure listed, age within range (inclusive lower, exclusive upper), gender match, preauthorization if required. Provide a step-by-step analysis.\n\n"
        "Example APPROVE:\n"
        "Coverage Review: Checked diagnosis match (yes), procedure listed (yes), age 30 in 25-40 (yes), gender match (yes), preauthorization required (no).\n"
        "Summary of Findings: All criteria met.\n"
        "Final Decision: APPROVE\n"
        "Reason: The claim for Procedure X (CPT 12345) has been approved because all required conditions were satisfied — the diagnosis matched, the procedure is covered, the patient's age falls within the allowed range, the gender matches policy requirements, and preauthorization was either not required or was properly obtained.\n\n"
 
        "Example ROUTE FOR REVIEW:\n"
        "Coverage Review: Checked diagnosis match (no), procedure listed (yes), age 50 in 25-40 (no), gender match (yes), preauthorization required (yes) but not obtained.\n"
        "Summary of Findings: Diagnosis mismatch, age out of range, preauthorization missing.\n"
        "Final Decision: ROUTE FOR REVIEW\n"
        "Reason: The claim for Procedure X (CPT 12345) cannot be automatically approved due to diagnosis mismatch (ICD-10 A01) and age 50 outside 25-40.\n\n"
 
        "Patient Summary: {record_summary}\n"
        "Policy Summary: {policy_summary}\n\n"
 
        "Output must strictly follow:\n"
        "Coverage Review: [Step-by-step checks]\n"
        "Summary of Findings: [Criteria met/not met]\n"
        "Final Decision: [APPROVE or ROUTE FOR REVIEW]\n"
        "Reason: [Explanation with procedure name, CPT code, and specific failed/met criteria]"
    )
 
    response = chat_client.invoke(prompt.format(
        record_summary=record_summary,
        policy_summary=policy_summary
    ))
    logging.info(f"Claim Coverage for record: {response.content}")
    return response.content

 

## 6. ReAct Agent Configuration
class AgentState(TypedDict):
    input: str
    chat_history: List[HumanMessage]
    agent_outcome: Union[str, Dict]

system_prompt = """
You are a claim approval agent. Use tools in this order:
1. summarize_patient_record
2. summarize_policy_guideline
3. check_claim_coverage

After step 3, stop. Extract and return only the final decision in this format:
Decision: [APPROVE or ROUTE FOR REVIEW]
Reason: [Short explanation from check_claim_coverage]

Do not loop or repeat tools.
"""

react_agent = create_react_agent(chat_client, tools=[summarize_patient_record, summarize_policy_guideline, check_claim_coverage])


## 7. Rate Limit Handling
def parse_retry_time(error_message: str) -> float:
    match = re.search(r'Please try again in (\d+h)?(\d+m)?(\d*\.?\d*s)?', error_message)
    if not match:
        return 60.0
    hours = match.group(1) or '0h'
    minutes = match.group(2) or '0m'
    seconds = match.group(3) or '0s'
    total_seconds = sum(int(t[:-1]) * (3600 if 'h' in t else 60 if 'm' in t else 1) for t in [hours, minutes, seconds] if t)
    return max(total_seconds, 1.0)
 
def process_record(record, max_retries=3, initial_delay=30):
    try:
        policy_id = record.get('insurance_policy_id')
        policy = insurance_policies_map.get(policy_id, {})
        if not policy:
            return f"Decision: ROUTE FOR REVIEW\nReason: No policy found for policy_id {policy_id}."
 
        record_str = json.dumps(record)
        inputs = {"messages": [HumanMessage(content=record_str)], "config": {"recursion_limit": 50}}
 
        for attempt in range(max_retries):
            try:
                response = react_agent.invoke(inputs)
                for msg in reversed(response['messages']):
                    if hasattr(msg, 'name') and msg.name == 'check_claim_coverage':
                        content = msg.content.strip()
                        lines = content.split('\n')
                        # Log or print the raw response content
                        print(f"🔍 Raw output from check_claim_coverage:\n{content}")
                        logging.info(f"check_claim_coverage content: {content}")
 
                        # Flexible search for decision and reason
                        decision_line = next((line for line in content.split('\n') if 'Decision:' in line), None)
                        reason_line = next((line for line in content.split('\n') if 'Reason:' in line or 'Explanation:' in line), None)
 
                       
                        if decision_line and reason_line:
                            decision = decision_line.replace('Final Decision: ', '').strip()
                            reason = reason_line.replace('Reason: ', '').strip()
                            return f"Decision: {decision}\n{reason}"
                        logging.warning(f"No valid decision/reason found in response for {record.get('patient_id')}")
                        return (
                            "Decision: ROUTE FOR REVIEW\n"
                            f"Reason: Unable to determine claim status for patient {record.get('patient_id')}."
                        )
                logging.warning(f"No check_claim_coverage response for {record.get('patient_id')}")
                return (
                    "Decision: ROUTE FOR REVIEW\n"
                    f"Reason: No coverage check completed for patient {record.get('patient_id')}."
                )
            except RateLimitError as e:
                retry_time = parse_retry_time(str(e))
                logging.warning(f"Rate limit error for {record.get('patient_id')}, attempt {attempt + 1}/{max_retries}: {e}. Retrying in {retry_time}s")
                print(f"Rate limit error for {record.get('patient_id')}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_time}s...")
                if attempt < max_retries - 1:
                    time.sleep(retry_time)
                else:
                    return (
                        "Decision: ROUTE FOR REVIEW\n"
                        f"Reason: Max retries reached due to rate limit for patient {record.get('patient_id')}."
                    )
            except Exception as e:
                logging.error(f"Error processing record {record.get('patient_id')}: {e}")
                print(f"Error processing record {record.get('patient_id')}: {e}")
                return f"Decision: ROUTE FOR REVIEW\nReason: Processing error for patient {record.get('patient_id')}."
    except Exception as e:
        logging.error(f"Unexpected error processing record {record.get('patient_id')}: {e}")
        print(f"Unexpected error processing record {record.get('patient_id')}: {e}")
        return f"Decision: ROUTE FOR REVIEW\nReason: Unexpected error for patient {record.get('patient_id')}."

## 8. Validation and Testing
batch_size = 5
validation_responses = []
initial_delay=15
print("Running validation records in batches...")
for i in range(0, len(validation_records), batch_size):
    batch = validation_records[i:i + batch_size]
    for record in batch:
        patient_id = record.get('patient_id', 'Unknown')
        print(f"Processing patient: {patient_id}")
        response = process_record(record)
        validation_responses.append({'patient_id': patient_id, 'generated_response': response})
        print(f"Agent response: {response}")
        print("-" * 50)
        time.sleep(initial_delay)
    if i + batch_size < len(validation_records):
        print(f"Pausing for 60 seconds after batch {i // batch_size + 1}...")
        time.sleep(60)
 
validation_df = pd.DataFrame(validation_responses)
validation_df.to_csv('./Data/validation_outputs.csv', index=False)
print("Validation outputs saved to './Data/validation_outputs.csv'.")
 
if validation_reference_results and validation_responses:
    matches = 0
    for resp in validation_responses:
        patient_id = resp['patient_id']
        generated_decision_line = resp['generated_response'].split('\n')[0].strip().lower()
        reference_decision_line = (
            validation_reference_results.get(patient_id, {})
            .get('reference_response', '')
            .split('\n')[0]
            .strip()
            .lower()
        )
 
        if generated_decision_line == reference_decision_line:
            matches += 1
        else:
            print(f"❌ Mismatch for {patient_id}: Generated → {generated_decision_line}, Expected → {reference_decision_line}")
 
    accuracy = matches / len(validation_responses) * 100
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")
 
    if accuracy < 100:
        logging.warning(f"Validation accuracy {accuracy:.2f}% is below 100%. Check logs for details.")
        print("Warning: Validation accuracy is below 100%. Review 'agent_debug.log' for mismatches.")
 
test_responses = []
print("Running test records in batches...")
for i in range(0, len(test_records), batch_size):
    batch = test_records[i:i + batch_size]
    for record in batch:
        patient_id = record.get('patient_id', 'Unknown')
        print(f"Processing patient: {patient_id}")
        response = process_record(record)
        test_responses.append({'patient_id': patient_id, 'generated_response': response})
        print(f"Agent response: {response}")
        print("-" * 50)
        time.sleep(initial_delay)
    if i + batch_size < len(test_records):
        print(f"Pausing for 60 seconds after batch {i // batch_size + 1}...")
        time.sleep(60)

test_df = pd.DataFrame(test_responses)
test_df.to_csv('./submission.csv', index=False)
print("Submission file './submission.csv' generated successfully.")

print("\nSample Validation Outputs:")
if validation_df.empty:
    print("No validation outputs to display.")
else:
    print(validation_df.head())