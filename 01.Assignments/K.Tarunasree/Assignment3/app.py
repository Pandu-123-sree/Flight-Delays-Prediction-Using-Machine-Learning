from flask import Flask, request, render_template, redirect
import sqlite3

app = Flask(__name__)

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect('tasks.db')  # Create or open the database file
    cursor = conn.cursor()
    
    # Create tasks table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            status INTEGER NOT NULL DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database when the application starts
init_db()

# Route to display the list of tasks
@app.route('/')
def home():
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()
    print(f"Tasks fetched: {tasks}")  # Debugging output
    return render_template('home.html', tasks=tasks)

# Route to create a new task
@app.route('/create', methods=['GET', 'POST'])
def create_task():
    if request.method == 'POST':
        description = request.form['description']
        conn = sqlite3.connect('tasks.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tasks (description) VALUES (?)", (description,))
        conn.commit()
        conn.close()
        return redirect('/')
    return render_template('create.html')

# Route to update an existing task
@app.route('/update/<int:task_id>', methods=['GET', 'POST'])
def update_task(task_id):
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        description = request.form['description']
        status = request.form.get('status', 0)  # 0 for incomplete, 1 for complete
        cursor.execute("UPDATE tasks SET description = ?, status = ? WHERE id = ?", (description, status, task_id))
        conn.commit()
        conn.close()
        return redirect('/')
    
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    task = cursor.fetchone()
    conn.close()
    
    if task is None:
        return "Task not found", 404  # Handle the case where the task doesn't exist
    
    return render_template('update.html', task=task)

# Route to delete a task
@app.route('/delete/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    conn = sqlite3.connect('tasks.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return redirect('/')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
