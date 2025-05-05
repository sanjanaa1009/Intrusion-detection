import google.generativeai as genai
import streamlit as st
import os
from typing import Dict, Any, List, Optional

class SecurityChatbot:
    """
    AI-powered security chatbot using Google's Gemini API
    """
    def __init__(self):
        """Initialize the chatbot with the Gemini API"""
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            
            # Configure the model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 0,
                "max_output_tokens": 2048,
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
            
            # Initialize model
            try:
                self.model = genai.GenerativeModel(
                    model_name="gemini-pro",
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                self.chat = self.model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": ["You are a cybersecurity expert assistant for an enterprise intrusion detection system. Keep your answers focused on security topics and provide actionable insights for security teams. You have access to network logs, user activity data, and anomaly detection results. Provide concise, security-focused answers to user questions."]
                        },
                        {
                            "role": "model",
                            "parts": ["I'm your cybersecurity expert assistant for the enterprise intrusion detection system. I'll focus on providing actionable security insights based on your network logs, user activities, and anomaly detection results. How can I help secure your environment today?"]
                        }
                    ]
                )
                self.available = True
            except Exception as e:
                self.available = False
                self.error_message = str(e)
        else:
            self.available = False
            self.error_message = "Gemini API key not configured"
    
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
        if not self.available:
            return f"Sorry, the security assistant is not available: {self.error_message}"
        
        try:
            # Add context to the question if available
            if context:
                enhanced_question = f"Context: {context}\n\nQuestion: {question}"
            else:
                enhanced_question = question
            
            # Get response from Gemini
            response = self.chat.send_message(enhanced_question)
            return response.text
        except Exception as e:
            return f"Sorry, there was an error generating a response: {str(e)}"


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
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Ask security-related questions about your enterprise environment, threat detection, vulnerabilities, 
    or best practices. The AI assistant will provide tailored recommendations and actionable insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I'm your AI security assistant. How can I help you analyze your security data or address enterprise security concerns?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Function to generate response
    def generate_response(prompt):
        # Get context from session state (if available)
        context = {}
        if "results" in st.session_state and st.session_state.results is not None:
            context["network_logs"] = f"Found {sum(st.session_state.results['prediction'] != 0)} threats out of {len(st.session_state.results)} logs analyzed"
        
        if "user_behavior_results" in st.session_state and st.session_state.user_behavior_results is not None:
            anomalous_users = st.session_state.user_behavior_results[st.session_state.user_behavior_results['prediction'] == -1]
            context["user_behavior"] = f"Found {len(anomalous_users)} anomalous user behaviors"
        
        if "unknown_threat_results" in st.session_state and st.session_state.unknown_threat_results is not None:
            unknown_threats = st.session_state.unknown_threat_results[st.session_state.unknown_threat_results['category'] != 'Normal']
            context["unknown_threats"] = f"Found {len(unknown_threats)} unknown threat patterns"
        
        # Get response from chatbot
        return chatbot.get_response(prompt, context)
    
    # Message input
    if not chatbot.available:
        st.warning(f"Security Assistant is not available: {chatbot.error_message}")
        if st.button("Configure Gemini API Key"):
            # Will trigger the secrets dialog to set the API key
            pass
    else:
        if prompt := st.chat_input("Ask about security events, threat analysis, or recommendations..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing security data..."):
                    response = generate_response(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})