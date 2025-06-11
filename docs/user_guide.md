# Prometheus User Guide

## Introduction

Prometheus is an intelligent assistant designed to help you understand your insurance policies. It can answer questions about your coverage, benefits, limitations, and more by analyzing your policy documents. This guide will help you get the most out of the system.

## Getting Started

### Accessing the System

You can access Prometheus through:

1. **Web Interface**: Navigate to the provided URL in your web browser
2. **API Integration**: For developers integrating with the system

### Authentication

To use Prometheus, you'll need to provide your National ID number. This allows the system to retrieve your specific policy documents.

## Using the Web Interface

### Home Screen

The home screen provides:

- A search box for entering your questions
- A field for your National ID
- Suggested questions based on your policies

### Asking Questions

1. Enter your National ID in the designated field
2. Type your question in the search box
3. Click "Submit" or press Enter

Example questions:
- "What is my dental coverage limit?"
- "Am I covered for emergency treatment abroad?"
- "What is my co-payment percentage for outpatient services?"

### Understanding Responses

Responses include:
- A direct answer to your question
- Supporting information from your policy documents
- Source references showing where the information was found
- Monetary amounts formatted as **QR X,XXX**
- Percentage values formatted as **XX%**

### Suggested Questions

The system analyzes your policy documents and suggests relevant questions you might want to ask. These appear:
- When you first enter your National ID
- After you receive an answer to a question
- In the sidebar of the interface

Click on any suggested question to submit it automatically.

## Advanced Features

### Policy Details

You can view detailed information about your policies by clicking on the "Policy Details" button. This shows:
- Policy numbers
- Insurance company names
- Coverage periods
- Annual limits
- Geographical coverage areas

### Document Exploration

To explore your policy documents directly:
1. Click on a source reference in any answer
2. The system will highlight the relevant section in your policy document
3. You can navigate through the document to read related information

### Follow-up Questions

The system understands context, so you can ask follow-up questions without repeating all the details. For example:
1. Ask: "What is my coverage for dental treatment?"
2. Then follow up with: "What about orthodontic treatment?"

## Tips for Effective Questions

### Be Specific

More specific questions yield more precise answers:

| Instead of | Ask |
|------------|-----|
| "What's my coverage?" | "What's my coverage for maternity services?" |
| "How much do I pay?" | "What's my co-payment for specialist consultations?" |

### Include Key Terms

Include key insurance terms in your questions:

- Coverage limits
- Deductibles
- Co-payments
- Pre-approvals
- Network providers
- Exclusions
- Benefits

### Question Types

Prometheus can answer various types of questions:

1. **Factual questions** about your policy:
   - "What is my annual coverage limit?"
   - "Is physiotherapy covered under my policy?"

2. **Procedural questions** about using your insurance:
   - "How do I submit a claim for reimbursement?"
   - "Do I need pre-approval for an MRI scan?"

3. **Comparative questions** about coverage options:
   - "What's the difference between in-network and out-of-network coverage?"
   - "How does my dental coverage compare to my vision coverage?"

## Troubleshooting

### No Answer Found

If the system cannot find an answer:
1. Try rephrasing your question
2. Use more specific terms from your policy
3. Check if the topic is covered in your policy documents

### Incorrect Information

If you believe the information is incorrect:
1. Click on the source reference to verify the original text
2. Contact your insurance provider for clarification
3. Provide feedback through the feedback button

### System Errors

If you encounter system errors:
1. Refresh the page and try again
2. Ensure your National ID is entered correctly
3. Contact support if the problem persists

## API Integration

For developers integrating with the Prometheus API:

### Query Endpoint

```
POST /api/query
```

**Request:**
```json
{
  "question": "What is my dental coverage limit?",
  "national_id": "12345678901"
}
```

**Response:**
```json
{
  "answer": "Your dental coverage has an annual limit of QR 2,000...",
  "sources": [
    {
      "content": "Dental treatment is covered up to QR 2,000 per year...",
      "source": "policy_document.pdf",
      "score": 0.92
    }
  ]
}
```

### Suggestions Endpoint

```
POST /api/suggestions
```

**Request:**
```json
{
  "national_id": "12345678901"
}
```

**Response:**
```json
{
  "questions": [
    "What is my annual coverage limit?",
    "Are dental procedures covered under my policy?",
    "What is the co-payment percentage for outpatient services?"
  ]
}
```

## Privacy and Security

Prometheus takes your privacy seriously:

- Your policy documents are accessed securely
- Queries are processed within the system and not shared with third parties
- National ID is used only for retrieving your specific policy information
- The system does not store your questions or create user profiles

## Getting Help

If you need assistance using Prometheus:

- Click the "Help" button in the interface
- Contact your insurance provider's customer service
- Email support at [health.claims@dig.qa](mailto:health.claims@dig.qa)

## Glossary of Insurance Terms

| Term | Definition |
|------|------------|
| Annual Limit | The maximum amount your insurance will pay for covered services in a policy year |
| Co-payment | A fixed amount you pay for a covered health care service |
| Deductible | The amount you pay before your insurance begins to pay |
| Exclusion | A condition or treatment not covered by your insurance policy |
| Network Provider | A healthcare provider contracted with your insurance company |
| Pre-approval | Authorization required from your insurance before receiving certain services |
| Premium | The amount you pay for your health insurance every month |
| Reimbursement | Payment from your insurance company for covered expenses you've paid |
