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


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/denoise_log.txt':
            # 返回日志文件内容
            try:
                with open('/home/yzx/LLaDA/denoise_log.txt', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, "File not found")
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
        
        elif parsed_path.path == '/api/steps':
            # 返回解析后的步骤数据
            try:
                with open('/home/yzx/LLaDA/denoise_log.txt', 'r', encoding='utf-8') as f:
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
                with open('/home/yzx/LLaDA/yzx_test/denoise_visualizer.html', 'r', encoding='utf-8') as f:
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
        """解析日志内容为步骤数组"""
        steps = []
        lines = content.split('\n')
        current_step = None
        collecting_result = False
        result_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 匹配步骤信息
            if line.startswith('Step'):
                # 如果已经有当前步骤，保存它
                if current_step is not None:
                    # 保存收集的结果
                    if result_lines:
                        current_step['result'] = ' '.join(result_lines)
                        result_lines = []
                    steps.append(current_step)
                
                # 解析步骤信息
                step_match = re.match(r'Step (\d+)/(\d+) \(Block (\d+)\), Transferred tokens: (\d+)', line)
                if step_match:
                    current_step = {
                        'stepNum': int(step_match.group(1)),
                        'totalSteps': int(step_match.group(2)),
                        'block': int(step_match.group(3)),
                        'transferredTokens': int(step_match.group(4)),
                        'result': ''
                    }
                    collecting_result = False
            # 匹配结果开始
            elif line.startswith('result:') and current_step:
                # 提取结果内容
                result = line[len('result:'):].strip()
                result_lines = [result]
                collecting_result = True
            # 收集结果的后续行
            elif collecting_result and current_step:
                # 只收集非空行，跳过分隔符
                if line and not line.startswith('---'):
                    result_lines.append(line)
        
        # 如果还有未保存的步骤，保存它
        if current_step and current_step.get('result'):
            steps.append(current_step)
        
        # 按步骤号排序
        steps.sort(key=lambda x: (x['block'], x['stepNum']))
        
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
    DIRECTORY = "/home/yzx/LLaDA"
    
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