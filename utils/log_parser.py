import pandas as pd
import re
import datetime
from typing import List, Dict, Union, Tuple, Optional
import ipaddress

class LogParser:
    """
    Utility class for parsing various log formats into structured data
    """
    
    @staticmethod
    def is_valid_ip(ip_str: str) -> bool:
        """
        Check if a string is a valid IP address
        """
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def parse_common_log_format(log_lines: List[str]) -> pd.DataFrame:
        """
        Parse Apache/Nginx common log format
        
        Format: %h %l %u %t "%r" %>s %b
        Example: 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
        
        Returns DataFrame with columns:
        - ip_address, user_id, timestamp, request, status_code, bytes_sent
        """
        # Regex pattern for common log format
        pattern = r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d+) (\S+)'
        
        records = []
        
        for line in log_lines:
            match = re.match(pattern, line)
            if match:
                ip, _, user_id, date_str, request, status, bytes_sent = match.groups()
                
                # Parse timestamp
                try:
                    timestamp = datetime.datetime.strptime(
                        date_str, "%d/%b/%Y:%H:%M:%S %z"
                    )
                except ValueError:
                    timestamp = None
                
                # Parse request components
                request_parts = request.split(" ")
                method = request_parts[0] if len(request_parts) > 0 else ""
                path = request_parts[1] if len(request_parts) > 1 else ""
                protocol = request_parts[2] if len(request_parts) > 2 else ""
                
                records.append({
                    'ip_address': ip,
                    'user_id': user_id if user_id != '-' else None,
                    'timestamp': timestamp,
                    'request': request,
                    'method': method,
                    'path': path,
                    'protocol': protocol,
                    'status_code': int(status) if status.isdigit() else 0,
                    'bytes_sent': int(bytes_sent) if bytes_sent.isdigit() else 0
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def parse_combined_log_format(log_lines: List[str]) -> pd.DataFrame:
        """
        Parse Apache/Nginx combined log format
        
        Format: %h %l %u %t "%r" %>s %b "%{Referer}i" "%{User-agent}i"
        Example: 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://example.com/start.html" "Mozilla/4.08 [en] (Win98; I ;Nav)"
        
        Returns DataFrame with columns:
        - ip_address, user_id, timestamp, request, status_code, bytes_sent, referer, user_agent
        """
        # Regex pattern for combined log format
        pattern = r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d+) (\S+) "(.*?)" "(.*?)"'
        
        records = []
        
        for line in log_lines:
            match = re.match(pattern, line)
            if match:
                ip, _, user_id, date_str, request, status, bytes_sent, referer, user_agent = match.groups()
                
                # Parse timestamp
                try:
                    timestamp = datetime.datetime.strptime(
                        date_str, "%d/%b/%Y:%H:%M:%S %z"
                    )
                except ValueError:
                    timestamp = None
                
                # Parse request components
                request_parts = request.split(" ")
                method = request_parts[0] if len(request_parts) > 0 else ""
                path = request_parts[1] if len(request_parts) > 1 else ""
                protocol = request_parts[2] if len(request_parts) > 2 else ""
                
                records.append({
                    'ip_address': ip,
                    'user_id': user_id if user_id != '-' else None,
                    'timestamp': timestamp,
                    'request': request,
                    'method': method,
                    'path': path,
                    'protocol': protocol,
                    'status_code': int(status) if status.isdigit() else 0,
                    'bytes_sent': int(bytes_sent) if bytes_sent.isdigit() else 0,
                    'referer': referer if referer != '-' else None,
                    'user_agent': user_agent
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def parse_syslog_format(log_lines: List[str]) -> pd.DataFrame:
        """
        Parse syslog format logs
        
        Format: <timestamp> <hostname> <process>[<pid>]: <message>
        Example: Jan 1 00:00:00 localhost sshd[12345]: Failed password for invalid user test from 192.168.1.1 port 58803 ssh2
        
        Returns DataFrame with columns:
        - timestamp, hostname, process, pid, message
        """
        # Regex pattern for syslog format
        pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+)(?:\[(\d+)\])?: (.*)'
        
        records = []
        
        for line in log_lines:
            match = re.match(pattern, line)
            if match:
                date_str, hostname, process, pid, message = match.groups()
                
                # Parse timestamp (add current year)
                current_year = datetime.datetime.now().year
                try:
                    timestamp = datetime.datetime.strptime(
                        f"{current_year} {date_str}", "%Y %b %d %H:%M:%S"
                    )
                except ValueError:
                    timestamp = None
                
                # Extract IP addresses from the message
                ip_addresses = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', message)
                ip_address = next((ip for ip in ip_addresses if LogParser.is_valid_ip(ip)), None)
                
                # Extract username if present (common in auth logs)
                username_match = re.search(r'user (\S+)', message)
                username = username_match.group(1) if username_match else None
                
                records.append({
                    'timestamp': timestamp,
                    'hostname': hostname,
                    'process': process,
                    'pid': int(pid) if pid and pid.isdigit() else None,
                    'message': message,
                    'ip_address': ip_address,
                    'user_id': username
                })
        
        return pd.DataFrame(records)
    
    @staticmethod
    def parse_json_logs(log_lines: List[str]) -> pd.DataFrame:
        """
        Parse JSON format logs
        
        Example: {"timestamp": "2023-01-01T00:00:00Z", "level": "info", "message": "User login", "user": "admin", "ip": "192.168.1.1"}
        
        Returns DataFrame with parsed JSON data
        """
        import json
        
        records = []
        
        for line in log_lines:
            try:
                # Try to parse as JSON
                data = json.loads(line)
                
                # Standardize some common fields
                if 'timestamp' in data and isinstance(data['timestamp'], str):
                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                    except:
                        pass
                
                # Map some common field names to our standard names
                if 'user' in data and 'user_id' not in data:
                    data['user_id'] = data['user']
                
                if 'ip' in data and 'ip_address' not in data:
                    data['ip_address'] = data['ip']
                
                records.append(data)
            except json.JSONDecodeError:
                # Not a valid JSON line, skip
                continue
        
        # Create DataFrame and handle case when no records were parsed
        if records:
            return pd.DataFrame(records)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def detect_log_format(log_sample: str) -> str:
        """
        Detect the format of a log line
        
        Returns: 'common', 'combined', 'syslog', 'json', or 'unknown'
        """
        # Check for JSON format first
        if log_sample.strip().startswith('{') and log_sample.strip().endswith('}'):
            try:
                import json
                json.loads(log_sample)
                return 'json'
            except:
                pass
        
        # Check for combined log format (has more parts than common log format)
        combined_pattern = r'\S+ \S+ \S+ \[.*?\] ".*?" \d+ \S+ ".*?" ".*?"'
        if re.match(combined_pattern, log_sample):
            return 'combined'
        
        # Check for common log format
        common_pattern = r'\S+ \S+ \S+ \[.*?\] ".*?" \d+ \S+'
        if re.match(common_pattern, log_sample):
            return 'common'
        
        # Check for syslog format
        syslog_pattern = r'\w+\s+\d+\s+\d+:\d+:\d+\s+\S+\s+\S+(?:\[\d+\])?: .*'
        if re.match(syslog_pattern, log_sample):
            return 'syslog'
        
        # Unknown format
        return 'unknown'
    
    @staticmethod
    def parse_logs(log_lines: List[str]) -> Tuple[pd.DataFrame, str]:
        """
        Parse logs of any supported format
        
        Returns:
        - DataFrame with parsed logs
        - String indicating the detected format
        """
        if not log_lines:
            return pd.DataFrame(), 'unknown'
        
        # Detect format using the first non-empty line
        first_line = next((line for line in log_lines if line.strip()), "")
        detected_format = LogParser.detect_log_format(first_line)
        
        # Parse according to detected format
        if detected_format == 'common':
            return LogParser.parse_common_log_format(log_lines), 'common'
        elif detected_format == 'combined':
            return LogParser.parse_combined_log_format(log_lines), 'combined'
        elif detected_format == 'syslog':
            return LogParser.parse_syslog_format(log_lines), 'syslog'
        elif detected_format == 'json':
            return LogParser.parse_json_logs(log_lines), 'json'
        else:
            # Try all parsers and see which one produces the most valid records
            results = []
            
            # Try common log format
            df_common = LogParser.parse_common_log_format(log_lines)
            if not df_common.empty:
                results.append((df_common, 'common', len(df_common)))
            
            # Try combined log format
            df_combined = LogParser.parse_combined_log_format(log_lines)
            if not df_combined.empty:
                results.append((df_combined, 'combined', len(df_combined)))
            
            # Try syslog format
            df_syslog = LogParser.parse_syslog_format(log_lines)
            if not df_syslog.empty:
                results.append((df_syslog, 'syslog', len(df_syslog)))
            
            # Try JSON format
            df_json = LogParser.parse_json_logs(log_lines)
            if not df_json.empty:
                results.append((df_json, 'json', len(df_json)))
            
            # Return the format that produced the most records
            if results:
                results.sort(key=lambda x: x[2], reverse=True)
                return results[0][0], results[0][1]
            
            # If all parsers failed, return empty DataFrame
            return pd.DataFrame({'message': log_lines}), 'raw'

    @staticmethod
    def extract_features_for_anomaly_detection(df: pd.DataFrame, log_format: str) -> pd.DataFrame:
        """
        Extract features for anomaly detection from parsed logs
        
        Returns DataFrame with features suitable for anomaly detection
        """
        if df.empty:
            return pd.DataFrame()
        
        # Common features across all log formats
        features = []
        
        # Process based on log format
        if log_format in ['common', 'combined']:
            # Group by IP address
            ip_groups = df.groupby('ip_address')
            
            for ip, group in ip_groups:
                # Skip if IP is invalid
                if not ip or not LogParser.is_valid_ip(ip):
                    continue
                
                # Basic features
                record = {
                    'ip_address': ip,
                    'user_id': ip,  # Use IP as user_id if no actual user_id
                    'request_count': len(group),
                    'unique_paths': group['path'].nunique(),
                    'error_rate': (group['status_code'] >= 400).mean(),
                    'bytes_total': group['bytes_sent'].sum(),
                    'avg_bytes': group['bytes_sent'].mean(),
                }
                
                # Add time-based features if timestamp is available
                if 'timestamp' in group.columns and not group['timestamp'].isnull().all():
                    timestamps = group['timestamp'].dropna()
                    if len(timestamps) > 1:
                        # Calculate time intervals between requests
                        sorted_times = sorted(timestamps)
                        intervals = [(sorted_times[i] - sorted_times[i-1]).total_seconds() 
                                    for i in range(1, len(sorted_times))]
                        
                        record.update({
                            'min_interval': min(intervals) if intervals else 0,
                            'avg_interval': sum(intervals) / len(intervals) if intervals else 0,
                            'request_rate': len(intervals) / (max(timestamps) - min(timestamps)).total_seconds() 
                                            if (max(timestamps) - min(timestamps)).total_seconds() > 0 else 0
                        })
                
                # Add method distribution
                if 'method' in group.columns:
                    for method in ['GET', 'POST', 'PUT', 'DELETE']:
                        record[f'method_{method}_rate'] = (group['method'] == method).mean()
                
                # Add status code distribution
                if 'status_code' in group.columns:
                    for status_range in [200, 300, 400, 500]:
                        record[f'status_{status_range}_rate'] = ((group['status_code'] >= status_range) & 
                                                               (group['status_code'] < status_range + 100)).mean()
                
                features.append(record)
        
        elif log_format == 'syslog':
            # Group by IP address and process
            if 'ip_address' in df.columns and not df['ip_address'].isnull().all():
                groups = df.groupby(['ip_address', 'process'])
                group_key = ['ip_address', 'process']
            else:
                groups = df.groupby('process')
                group_key = ['process']
            
            for keys, group in groups:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                
                # Basic record
                record = dict(zip(group_key, keys))
                
                # Use first field as user_id if not available
                if 'user_id' not in record:
                    record['user_id'] = record.get(group_key[0], 'unknown')
                
                # Add basic metrics
                record.update({
                    'message_count': len(group),
                })
                
                # Add time-based features if timestamp is available
                if 'timestamp' in group.columns and not group['timestamp'].isnull().all():
                    timestamps = group['timestamp'].dropna()
                    if len(timestamps) > 1:
                        # Calculate time intervals between messages
                        sorted_times = sorted(timestamps)
                        intervals = [(sorted_times[i] - sorted_times[i-1]).total_seconds() 
                                    for i in range(1, len(sorted_times))]
                        
                        record.update({
                            'min_interval': min(intervals) if intervals else 0,
                            'avg_interval': sum(intervals) / len(intervals) if intervals else 0,
                            'message_rate': len(intervals) / (max(timestamps) - min(timestamps)).total_seconds() 
                                           if (max(timestamps) - min(timestamps)).total_seconds() > 0 else 0
                        })
                
                # Look for common patterns in messages
                if 'message' in group.columns:
                    # Count 'error', 'failed', 'warning' keywords
                    messages = ' '.join(group['message'].fillna('').astype(str)).lower()
                    record.update({
                        'error_count': messages.count('error'),
                        'failed_count': messages.count('failed'),
                        'warning_count': messages.count('warning'),
                        'critical_count': messages.count('critical'),
                    })
                
                features.append(record)
        
        elif log_format == 'json':
            # For JSON logs, we need to handle variable schema
            # Start by identifying a good grouping key - prefer user_id, ip_address, or any 'id' field
            potential_keys = ['user_id', 'ip_address', 'id', 'source']
            group_key = next((key for key in potential_keys if key in df.columns and not df[key].isnull().all()), None)
            
            if group_key:
                groups = df.groupby(group_key)
                
                for key, group in groups:
                    # Basic record
                    record = {
                        group_key: key,
                        'user_id': key if group_key == 'user_id' else key if group_key == 'ip_address' else None,
                        'event_count': len(group)
                    }
                    
                    # Add time-based features if timestamp exists
                    if 'timestamp' in group.columns and not group['timestamp'].isnull().all():
                        timestamps = pd.to_datetime(group['timestamp'], errors='coerce').dropna()
                        if len(timestamps) > 1:
                            # Calculate time intervals between events
                            sorted_times = sorted(timestamps)
                            intervals = [(sorted_times[i] - sorted_times[i-1]).total_seconds() 
                                        for i in range(1, len(sorted_times))]
                            
                            record.update({
                                'min_interval': min(intervals) if intervals else 0,
                                'avg_interval': sum(intervals) / len(intervals) if intervals else 0,
                                'event_rate': len(intervals) / (max(timestamps) - min(timestamps)).total_seconds() 
                                               if (max(timestamps) - min(timestamps)).total_seconds() > 0 else 0
                            })
                    
                    # Add level distribution if available
                    if 'level' in group.columns:
                        for level in ['info', 'warn', 'error', 'debug']:
                            level_count = (group['level'].astype(str).str.lower() == level).sum()
                            record[f'level_{level}_rate'] = level_count / len(group) if len(group) > 0 else 0
                    
                    # For all numerical columns, calculate stats
                    numeric_cols = group.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        if col not in [group_key, 'timestamp']:
                            values = group[col].dropna()
                            if len(values) > 0:
                                record[f'{col}_mean'] = values.mean()
                                record[f'{col}_max'] = values.max()
                    
                    features.append(record)
            else:
                # If no good grouping key, just create one record per row
                for _, row in df.iterrows():
                    record = row.to_dict()
                    
                    # Ensure user_id exists
                    if 'user_id' not in record or pd.isnull(record['user_id']):
                        record['user_id'] = record.get('ip_address', f"unknown_{_}")
                    
                    features.append(record)
        
        features_df = pd.DataFrame(features)
        
        # Fill NaN values with 0
        features_df = features_df.fillna(0)
        
        return features_df