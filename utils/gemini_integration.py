import os
import google.generativeai as genai
from typing import Dict, Optional

def initialize_gemini_api():
    """
    Initialize the Gemini API with the API key from environment variables
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            print(f"Error initializing Gemini API: {str(e)}")
            return False
    else:
        print("No Gemini API key found")
        return False

def get_gemini_recommendation(attack_type: str, log_data: Dict) -> str:
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
    try:
        # Check if Gemini API is initialized
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ Gemini API key not configured. Please configure it in the settings to get AI-powered recommendations."
        
        # Configure the model
        generation_config = {
            "temperature": 0.2,  # More deterministic for security recommendations
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Create a prompt for getting recommendations
        prompt = f"""
        As a cybersecurity expert, provide actionable recommendations for responding to this detected threat:
        
        Attack Type: {attack_type}
        
        Log Details:
        - Source IP: {log_data.get('src_ip', 'Unknown')}
        - Destination IP: {log_data.get('dst_ip', 'Unknown')}
        - Protocol: {log_data.get('proto', 'Unknown')}
        - Service: {log_data.get('service', 'Unknown')}
        
        Please provide:
        1. A brief explanation of this attack type
        2. Immediate actions to take
        3. Long-term mitigation strategies
        4. Recommended security tools or configurations
        
        Keep your response concise, practical, and focused on enterprise security best practices.
        """
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"⚠️ Could not generate recommendations: {str(e)}\n\nPlease make sure your Gemini API key is correctly configured."

def analyze_security_posture(logs_summary: Dict) -> str:
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
    try:
        # Check if Gemini API is initialized
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ Gemini API key not configured. Please configure it in the settings to get AI-powered analysis."
        
        # Configure the model
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 1024,
        }
        
        # Create a prompt for security posture analysis
        prompt = f"""
        As a cybersecurity expert, analyze this security detection summary and provide an assessment of the overall security posture:
        
        Detection Summary:
        - Total logs analyzed: {logs_summary.get('total_logs', 0)}
        - Detected threats: {logs_summary.get('detected_threats', 0)}
        - Detection rate: {logs_summary.get('detection_rate', '0%')}
        - Top attack categories: {logs_summary.get('top_categories', [])}
        - Anomalous user behaviors: {logs_summary.get('anomalous_users', 0)}
        - Unknown threat patterns: {logs_summary.get('unknown_threats', 0)}
        
        Please provide:
        1. An assessment of the current security posture based on these metrics
        2. Key risk areas that need immediate attention
        3. Recommended security improvements
        4. Suggested monitoring and alerting configurations
        
        Keep your response concise, practical, and focused on enterprise security best practices.
        """
        
        # Initialize model
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config
        )
        
        # Generate response
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"⚠️ Could not generate security posture analysis: {str(e)}\n\nPlease make sure your Gemini API key is correctly configured."