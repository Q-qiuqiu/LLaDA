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
        
        file_name='/home/yzx/LLaDA/denoise_log_128_128_128.txt'

        if parsed_path.path == '/denoise_log.txt':
            # 返回日志文件内容
            try:
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
        
        elif parsed_path.path == '/api/steps':
            # 返回解析后的步骤数据
            try:
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
        """解析日志内容为步骤数组（兼容：Block X Step i/j transferred=k 的格式）"""
        steps = []
        lines = content.splitlines()

        current_step = None
        result_lines = []

        # 兼容新格式：Block 1 Step 1/8 transferred=4
        new_header = re.compile(r'^Block\s+(\d+)\s+Step\s+(\d+)/(\d+)\s+transferred=(\d+)\s*$')
        # 兼容旧格式：Step 1/8 (Block 1), Transferred tokens: 4
        old_header = re.compile(r'^Step\s+(\d+)/(\d+)\s+\(Block\s+(\d+)\),\s+Transferred tokens:\s+(\d+)\s*$')

        def flush_current():
            nonlocal current_step, result_lines
            if current_step is None:
                return
            # 保留换行更适合展示（HTML 里用 pre-wrap）
            current_step['result'] = "\n".join(result_lines).rstrip()
            steps.append(current_step)
            current_step = None
            result_lines = []

        for raw_line in lines:
            line = raw_line.rstrip("\n")
            s = line.strip()

            # 分隔线：不作为内容写入
            if s.startswith('---') and set(s) <= set('-'):
                continue

            m = new_header.match(s)
            if m:
                flush_current()
                current_step = {
                    'stepNum': int(m.group(2)),
                    'totalSteps': int(m.group(3)),
                    'block': int(m.group(1)),
                    'transferredTokens': int(m.group(4)),
                    'result': ''
                }
                continue

            m = old_header.match(s)
            if m:
                flush_current()
                current_step = {
                    'stepNum': int(m.group(1)),
                    'totalSteps': int(m.group(2)),
                    'block': int(m.group(3)),
                    'transferredTokens': int(m.group(4)),
                    'result': ''
                }
                continue

            # 如果已经进入某个 step，就把后续行当作结果内容
            if current_step is not None:
                result_lines.append(line)

        # 别忘了最后一个 step
        flush_current()

        # 按 block + step 排序（你原来就这么做）
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
