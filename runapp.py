from app import app
app.jinja_env.cache = {}
app.run(host='127.0.0.1', port=5000, debug=True)
