import os
import json
import google.generativeai as genai
from datetime import datetime

def initialize_gemini_api():
    """
    Initialize the Gemini API with the API key from environment variables
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        return False

def get_gemini_recommendation(attack_type, log_data):
    """
    Get recommended actions for detected threats using Gemini API
    
    Parameters:
    -----------
    attack_type : str
        The type of attack detected
    log_data : dict
        The log data containing details about the attack
        
    Returns:
    --------
    str
        Recommended actions for the detected threat
    """
    # Check if API is available
    api_available = initialize_gemini_api()
    
    if not api_available:
        return "Gemini API not available. Please set the GEMINI_API_KEY environment variable."
    
    try:
        # Format the log data as a string
        log_data_str = json.dumps(log_data, indent=2)
        
        # Create the prompt
        prompt = f"""
        As a cybersecurity expert, provide a concise recommendation for handling the following 
        network security incident:
        
        Attack Type: {attack_type}
        
        Log Data:
        {log_data_str}
        
        Please provide:
        1. A brief explanation of this type of attack
        2. Immediate actions to take
        3. Long-term mitigation strategies
        
        Keep the response focused and practical.
        """
        
        # Generate response using Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

def analyze_security_posture(logs_summary):
    """
    Analyze overall security posture based on detection results
    
    Parameters:
    -----------
    logs_summary : dict
        Summary of detection results
        
    Returns:
    --------
    str
        Security posture analysis and recommendations
    """
    # Check if API is available
    api_available = initialize_gemini_api()
    
    if not api_available:
        return "Gemini API not available. Please set the GEMINI_API_KEY environment variable."
    
    try:
        # Format the logs summary as a string
        logs_summary_str = json.dumps(logs_summary, indent=2)
        
        # Create the prompt
        prompt = f"""
        As a cybersecurity expert, analyze the following intrusion detection results and provide
        an assessment of the overall security posture:
        
        Detection Summary:
        {logs_summary_str}
        
        Please provide:
        1. An overall assessment of the security posture
        2. Key areas of concern based on the detection results
        3. Prioritized recommendations for improving security
        
        Format the response in a clear, professional manner suitable for a security report.
        """
        
        # Generate response using Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        return f"Error generating security posture analysis: {str(e)}"
