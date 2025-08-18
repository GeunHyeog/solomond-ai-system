#!/usr/bin/env python3
"""
Simple web dashboard for SOLOMOND AI - Test interface
"""

from flask import Flask
import webbrowser
import threading
import time

app = Flask(__name__)

@app.route('/')
def dashboard():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>SOLOMOND AI Test Dashboard</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #f0f8ff; }
        .button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
        .status { background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🤖 SOLOMOND AI Test</h1>
    <div id="status" class="status">
        <p id="statusText">Ready to test</p>
    </div>
    
    <button class="button" onclick="testOllama()">🧪 Test Ollama</button>
    <button class="button" onclick="openModule1()">🏆 Module 1</button>
    <button class="button" onclick="openDashboard()">📊 Dashboard</button>
    
    <div id="results"></div>

    <script>
        async function testOllama() {
            document.getElementById('statusText').textContent = 'Testing Ollama...';
            try {
                const response = await fetch('http://localhost:8888/health');
                const data = await response.json();
                document.getElementById('statusText').innerHTML = 
                    data.ollama_connected ? 
                    `✅ Ollama OK - ${data.available_models.length} models` :
                    '❌ Ollama Failed';
            } catch (error) {
                document.getElementById('statusText').textContent = '❌ Server Error';
            }
        }
        
        function openModule1() {
            window.open('MODULE1_REAL_AI_ANALYSIS.html', '_blank');
        }
        
        function openDashboard() {
            window.open('dashboard.html', '_blank');
        }
    </script>
</body>
</html>
    '''

if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:8080')
    
    threading.Thread(target=open_browser, daemon=True).start()
    print("Test Dashboard: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
