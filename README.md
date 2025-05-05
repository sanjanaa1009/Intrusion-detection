# CyberSentry: Enterprise-Grade Intrusion Detection System

CyberSentry is a comprehensive intrusion detection system that combines three complementary ML models to detect known attacks, unknown threats, and anomalous user behavior in enterprise environments. It provides actionable insights with AI-powered security recommendations.

## Key Features

- **Known Attack Detection**: LGBM-based model to detect common attack patterns like DoS, Reconnaissance, Exploits
- **Zero-Day Threat Detection**: Isolation Forest to identify previously unknown attacks through pattern recognition
- **User Behavior Analysis**: Anomaly detection to spot suspicious user activities and lateral movements
- **Blockchain Verification**: Immutable storage of logs for compliance and audit
- **Security Assistant**: AI-powered chatbot for security insights and recommendations
- **Multi-format Log Support**: Parses Apache, Syslog, and JSON log formats

## Getting Started

1. Run the application: `streamlit run app.py`
2. The application will be available at http://localhost:5000
3. Navigate through different modules using the sidebar

## Required Data Formats

### Network Detection Module

CSV file with the following columns:
```
src_ip,dst_ip,proto,service,dur,sbytes,dbytes,sttl,dttl
```

Example:
```
src_ip,dst_ip,proto,service,dur,sbytes,dbytes,sttl,dttl
192.168.1.100,10.0.0.1,tcp,http,2.54,1460,4680,128,64
45.77.65.211,192.168.1.10,tcp,http,5.12,4280,1460,45,128
```

Additional columns like `attack_cat` and `label` are optional (used for training).

### User Behavior Analysis Module

CSV file with the following columns:
```
timestamp,user_id,ip_address,action,resource,duration,bytes_transferred,location,device_type,session_count,failed_attempts,risk_score
```

Example:
```
timestamp,user_id,ip_address,action,resource,duration,bytes_transferred,location,device_type,session_count,failed_attempts,risk_score
2023-07-01T08:30:00Z,john.doe,192.168.1.100,login,/dashboard,300,5000,New York,desktop,1,0,0.1
2023-07-01T21:30:00Z,john.doe,185.143.223.45,login,/dashboard,150,3500,Moscow,mobile,1,2,0.8
```

### Zero-Day Threat Detection Module

Any log format is supported. The system can parse:

1. **Apache/Nginx logs**:
```
192.168.1.100 - - [01/Jul/2023:15:42:31 +0000] "GET /index.html HTTP/1.1" 200 4523
```

2. **Syslog format**:
```
Jul 01 15:42:31 webserver sshd[12345]: Failed password for invalid user admin from 45.77.65.211 port 58803 ssh2
```

3. **JSON logs**:
```json
{"timestamp": "2023-07-01T15:42:31Z", "level": "info", "message": "User login", "user": "john", "ip": "192.168.1.100", "status": "success"}
```

4. **Generic text logs**:
```
Failed login attempt from IP 45.77.65.211 for user admin
```

## AI-powered Security Assistant

CyberSentry includes an AI-powered security assistant using the Gemini API. To use this feature:

1. Obtain a Gemini API key from Google AI Studio (https://ai.google.dev/)
2. Configure the API key in the application
3. Ask security-related questions to get actionable insights

## Sample Data

Sample data files are included in the `data/` directory:
- `sample_network_logs.csv`: Example network traffic data
- `sample_user_behavior.csv`: Example user behavior data
- `sample_log_entries.txt`: Example log data for Zero-Day detection