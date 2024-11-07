# Leadify AI - Intelligent Business Lead Generation

**University Project - Machine Learning Course**  
**Professor:** Dr. Mert  
**Academic Year:** 2023-2024  

### Students:
- **Hussein Hussein** (58301)  
- **Samia Boussaid** (60925)  

---

## Project Overview

Leadify AI is an intelligent business lead generation platform that combines **natural language processing**, **web scraping**, and **email validation** to help businesses find and validate potential leads. This project showcases the practical application of machine learning in business intelligence.

---

## Core Features

### 1. Natural Language Chatbot Interface
- **Custom-trained NER (Named Entity Recognition)** model for business and location extraction.  
- Intelligent conversation handling for multiple query types:
  - Business search queries.
  - Email validation requests.
  - General conversation.
- Built using **Transformers** library and custom training data.

### 2. Email Validation System
- Deep email validation with multiple verification methods:
  - MX record validation.
  - SMTP verification.
  - Regex pattern matching.
  - Disposable email detection.
- **Batch processing** capability for multiple email addresses.
- Real-time validation feedback.

### 3. Google Maps Integration
- Automated business search across regions.
- Location-based filtering.
- Business information extraction.

---

## Technical Architecture

### **Frontend**
- **Next.js** with TypeScript.  
- **Tailwind CSS** for styling.  
- Real-time chat interface.  
- Responsive design.  
- Server-side rendering.

### **Backend**
- **Node.js/Express** server.  
- Python ML service integration.  
- **RESTful API** architecture.  
- Worker threads for parallel processing.

### **Machine Learning Components**
- Custom **NER model** trained on a business-specific dataset.  
- Transformer-based architecture.  
- Python integration with Node.js through **child processes**.

---

## Key Integrations

The system integrates three main components:
1. **Chatbot Service (Python)** - Handles NLP and intent recognition.
2. **Email Validation Service (Node.js)** - Validates potential leads.
3. **Google Maps Scraper (Node.js)** - Gathers business information.

---

## Implementation Details

### **Chatbot Service**
- Processes natural language input.
- Detects user intent (search/validation/conversation).
- Returns structured responses for frontend rendering.

---

### **Email Validation**  

```javascript
export const validateSingleEmail = async (email) => {
  logger.info(`Validating email: ${email}`);
  try {
    const result = await validate({
      email: email,
      validateRegex: true,
      validateMx: true,
      validateTypo: false,
      validateDisposable: true,
      validateSMTP: true,
    });

    logger.info(`Validation result for ${email}:`, JSON.stringify(result, null, 2));

    return {
      isValid: result.valid,
      reason: result.reason || 'Valid email',
      validators: result.validators
    };
  } catch (error) {
    logger.error(`Error validating ${email}:`, error.message);
    return {
      isValid: false,
      reason: 'Error during validation',
      validators: {}
    };
  }
};
