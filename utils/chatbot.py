import google.generativeai as genai
import streamlit as st
import os
from typing import List, Dict, Any

class SecurityChatbot:
    """
    AI-powered security chatbot using Google's Gemini API
    """
    def __init__(self):
        """Initialize the chatbot with the Gemini API"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.is_configured = False
            return
            
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.is_configured = True
        
        # Set up a basic system prompt that defines the chatbot's role
        self.system_prompt = """
        You are CyberSentry, an expert cybersecurity assistant specialized in helping security analysts 
        interpret intrusion detection system alerts and logs. Your expertise includes:
        
        1. Network security and common attack patterns
        2. Log analysis and forensics
        3. Security best practices and remediation steps
        4. Threat intelligence
        
        Always be clear, concise, and informative. When analyzing potential security incidents, include:
        - Possible explanations for the observed behavior
        - Potential severity level and impact
        - Recommended next steps for investigation or remediation
        
        If you don't have enough information, ask targeted questions to better understand the context.
        """
        
        # Initialize chat session
        self.chat = self.model.start_chat(history=[
            {
                "role": "user",
                "parts": [self.system_prompt]
            },
            {
                "role": "model",
                "parts": ["I understand my role as CyberSentry. I'm ready to assist with cybersecurity questions, log analysis, and incident response guidance."]
            }
        ])
    
    def get_response(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Get a response from the chatbot
        
        Parameters:
        -----------
        question : str
            The user's question
        context : Dict
            Additional context to provide to the AI (e.g., recent alerts, logs, etc.)
            
        Returns:
        --------
        str
            The chatbot's response
        """
        if not self.is_configured:
            return ("I apologize, but I'm not fully configured. Please ensure the GEMINI_API_KEY "
                   "environment variable is set to use the AI chatbot features.")
        
        # Prepare the prompt with context if provided
        prompt = question
        if context:
            # Format context data as a string
            formatted_context = ""
            for key, value in context.items():
                formatted_context += f"--- {key} ---\n{value}\n\n"
            
            # Add context to the prompt
            prompt = f"Context information:\n{formatted_context}\n\nUser question: {question}"
        
        try:
            # Get response from Gemini
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again or check your API key configuration."

def initialize_chatbot() -> SecurityChatbot:
    """Initialize and return a chatbot instance"""
    return SecurityChatbot()

def chatbot_interface(chatbot: SecurityChatbot):
    """
    Render a chatbot interface in Streamlit
    
    Parameters:
    -----------
    chatbot : SecurityChatbot
        The chatbot instance
    """
    st.subheader("CyberSentry AI Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f"""
            <div style='background-color: #2E4053; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: white; font-weight: bold;'>You:</p>
                <p style='color: white;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <p style='color: #3498DB; font-weight: bold;'>CyberSentry:</p>
                <p style='color: white;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Input for user questions
    user_question = st.text_input("Ask a security question:", key="user_question")
    
    context = None
    if "detection_results" in st.session_state and st.session_state.detection_results is not None:
        # Add detection results as context
        context = {
            "recent_detection_results": str(st.session_state.detection_results.head(10))
        }
    
    if st.button("Send", key="send_question"):
        if user_question:
            # Add user question to chat history
            st.session_state.chat_history.append(("user", user_question))
            
            # Get response from chatbot
            response = chatbot.get_response(user_question, context)
            
            # Add chatbot response to chat history
            st.session_state.chat_history.append(("assistant", response))
            
            # Clear input
            st.session_state.user_question = ""
            
            # Rerun to update display
            st.rerun()
    
    # Add button to clear chat history
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()