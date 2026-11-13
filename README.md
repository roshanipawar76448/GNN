# GNN-Based Harassment Network Analysis

A Flask and MySQL application that represents user interactions as a graph, analyses connected communities and repeated targeting patterns, and compares GCN, GAT and GraphSAGE models with a Random Forest baseline.

## Project structure

```text
GNN-Based-Harassment-Network-Analysis/
├── app.py
├── analysis.py
├── database.py
├── requirements.txt
├── .env.example
├── README.md
├── database/
│   └── schema.sql
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

## Run in VS Code (Windows)

1. Install Python and MySQL Server.
2. Open this project folder in VS Code.
3. Open the terminal and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

4. Open `.env` and enter your MySQL password:

```ini
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=gnn_harassment
```

5. Make sure MySQL Server is running.
6. Start the website:

```bash
python app.py
```

7. Open http://127.0.0.1:5000 in your browser.

The application creates the configured database and required tables when it starts, provided the MySQL account has permission to create the database. If it does not, create the database using `database/schema.sql` or grant the required permission.

## Dataset

Upload a CSV file containing `source_user` and `target_user`. `is_abusive` is used to label interactions, `user_label` provides account-level labels for supervised model training, and `timestamp` enables temporal analysis.

## Important interpretation

Communities and repeated-targeting indicators identify patterns for further review; they do not prove malicious intent. Model scores depend on dataset quality, labels and evaluation design. The current implementation uses a seeded random node split and is a research prototype, not a production moderation system.
