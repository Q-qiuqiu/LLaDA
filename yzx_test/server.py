#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaDA 去噪过程可视化服务器
用于启动一个简单的HTTP服务器来查看denoise_log.txt的逐步结果
"""

import http.server
import socketserver
import os
import signal
import sys
from urllib.parse import urlparse, parse_qs
import json
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZER_HTML = os.path.join(BASE_DIR, 'denoise_visualizer.html')


def list_txt_files():
    files = []
    for name in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, name)
        if os.path.isfile(path) and name.lower().endswith('.txt'):
            files.append(name)
    return sorted(files)


def resolve_selected_file(query_params):
    files = list_txt_files()
    if not files:
        return None, "当前目录下没有可用的 txt 文件"

    requested = query_params.get('file', [None])[0]
    if requested:
        safe_name = os.path.basename(requested)
        if safe_name not in files:
            return None, f"File not found: {safe_name}"
        return os.path.join(BASE_DIR, safe_name), None

    return os.path.join(BASE_DIR, files[0]), None


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if parsed_path.path == '/denoise_log.txt':
            # 返回日志文件内容
            try:
                file_name, error = resolve_selected_file(query_params)
                if error:
                    self.send_error(404, error)
                    return
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, "File not found")
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")

        elif parsed_path.path == '/api/files':
            try:
                files = list_txt_files()
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(files, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        
        elif parsed_path.path == '/api/steps':
            # 返回解析后的步骤数据
            try:
                file_name, error = resolve_selected_file(query_params)
                if error:
                    self.send_error(404, error)
                    return
                with open(file_name, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # 解析日志内容
                steps = self.parse_log_content(log_content)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(steps, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        
        elif parsed_path.path == '/' or parsed_path.path == '/index.html':
            # 返回可视化界面HTML
            try:
                with open(VISUALIZER_HTML, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, "Visualization HTML file not found")
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        
        else:
            # 对于其他路径，尝试从当前目录提供文件
            super().do_GET()
    
    def parse_log_content(self, content):
        """按分隔线切块解析日志，并尽量从每块首行提取步骤元信息。"""

        def is_separator(line):
            stripped = line.strip()
            return len(stripped) >= 20 and set(stripped) == {'-'}

        def parse_metrics(extra_text):
            metrics = {}
            if not extra_text:
                return metrics
            for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)', extra_text):
                metrics[key] = value
            return metrics

        def to_int(value):
            if value is None:
                return None
            value = str(value).strip()
            if value.isdigit():
                return int(value)
            return None

        def split_into_chunks(raw_content):
            chunks = []
            current_lines = []
            for line in raw_content.splitlines():
                if is_separator(line):
                    if current_lines and any(item.strip() for item in current_lines):
                        chunks.append("\n".join(current_lines).strip("\n"))
                    current_lines = []
                    continue
                current_lines.append(line)

            if current_lines and any(item.strip() for item in current_lines):
                chunks.append("\n".join(current_lines).strip("\n"))
            return chunks

        def parse_chunk(chunk_text, default_index):
            lines = chunk_text.splitlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            if not lines:
                return None

            # 兼容：
            # Block 1 Step 1/128 selected=23 remain_ratio=1.0000
            block_header_re = re.compile(r'^Block\s+(\d+)\s+Step\s+(\d+)/(\d+)(?:\s+(.*))?\s*$')
            # 兼容：
            # Step 1/128 stage=TOOL k=2 remaining=128
            step_header_re = re.compile(r'^Step\s+(\d+)/(\d+)(?:\s+(.*))?\s*$')
            # 兼容旧格式：
            # Step 1/8 (Block 1), Transferred tokens: 4
            old_header_re = re.compile(r'^Step\s+(\d+)/(\d+)\s+\(Block\s+(\d+)\),\s+Transferred tokens:\s+(\d+)\s*$')

            header_index = None
            matched_header = None
            matched_header_type = None
            for index, line in enumerate(lines):
                header = line.strip()
                for header_type, pattern in (
                    ('block', block_header_re),
                    ('old', old_header_re),
                    ('step', step_header_re),
                ):
                    match = pattern.match(header)
                    if match:
                        header_index = index
                        matched_header = match
                        matched_header_type = header_type
                        break
                if matched_header is not None:
                    break

            step_num = default_index
            total_steps = None
            block = 1
            transferred_tokens = None
            selected_tokens = None
            remain_ratio = None
            body_lines = lines[:]

            if matched_header_type == 'block':
                metrics = parse_metrics(matched_header.group(4))
                step_num = int(matched_header.group(2))
                total_steps = int(matched_header.group(3))
                block = int(matched_header.group(1))
                transferred_tokens = to_int(metrics.get('transferred'))
                selected_tokens = to_int(metrics.get('selected'))
                remain_ratio = metrics.get('remain_ratio')
                body_lines = lines[header_index + 1:]
            elif matched_header_type == 'old':
                step_num = int(matched_header.group(1))
                total_steps = int(matched_header.group(2))
                block = int(matched_header.group(3))
                transferred_tokens = int(matched_header.group(4))
                body_lines = lines[header_index + 1:]
            elif matched_header_type == 'step':
                metrics = parse_metrics(matched_header.group(3))
                step_num = int(matched_header.group(1))
                total_steps = int(matched_header.group(2))
                transferred_tokens = to_int(metrics.get('transferred'))
                selected_tokens = to_int(metrics.get('selected'))
                remain_ratio = metrics.get('remain_ratio')
                body_lines = lines[header_index + 1:]

            return {
                'stepNum': step_num,
                'totalSteps': total_steps,
                'block': block,
                'transferredTokens': transferred_tokens,
                'selectedTokens': selected_tokens,
                'remainRatio': remain_ratio,
                'result': "\n".join(body_lines).rstrip()
            }

        chunks = split_into_chunks(content)
        steps = []

        for index, chunk in enumerate(chunks, start=1):
            step = parse_chunk(chunk, index)
            if step is not None:
                steps.append(step)

        if not steps:
            return []

        inferred_total = len(steps)
        for index, step in enumerate(steps, start=1):
            if step['stepNum'] is None:
                step['stepNum'] = index
            if step['totalSteps'] is None:
                step['totalSteps'] = inferred_total
            if step['block'] is None:
                step['block'] = 1

        return steps


def find_free_port(start_port=8080):
    """查找可用端口"""
    import socket
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许重用地址
                s.bind(('', port))
                return port
        except OSError:
            port += 1
            if port > start_port + 100:  # 防止无限循环
                raise OSError("Could not find free port")


def signal_handler(sig, frame):
    print('\n服务器正在关闭...')
    sys.exit(0)


def main():
    DIRECTORY = BASE_DIR
    
    # 更改工作目录
    os.chdir(DIRECTORY)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 查找可用端口
    PORT = find_free_port(34414)
    print(f"尝试使用端口: {PORT}")
    
    Handler = CustomHTTPRequestHandler
    
    # 创建服务器实例
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        # 设置允许重用地址
        httpd.allow_reuse_address = True
        
        print(f"LLaDA 去噪可视化服务器启动在端口 {PORT}")
        print(f"访问 http://localhost:{PORT} 查看可视化界面")
        print("按 Ctrl+C 停止服务器")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n服务器正在关闭...')
            httpd.shutdown()


if __name__ == "__main__":
    main()
