from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory, send_file
from celery_app import generate_embeddings_task, delete_files
import os
import pandas as pd
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process-matches', methods=['POST'])
def process_matches():
    try:
        file = request.files.get('file')  # Upload CSV file

        if not file:
            logging.error("No file uploaded.")
            return jsonify({"error": "No file uploaded"}), 400

        matches_df = pd.read_csv(file)

        # Validate required columns in the uploaded file
        required_columns = ["Guide Profile", "Student Profile"]
        missing_columns = [col for col in required_columns if col not in matches_df.columns]

        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns}")
            for col in missing_columns:
                matches_df[col] = 'N/A'  # Add default placeholder values

        # Placeholder for explanation generation (removed dependency on gpt_utils)
        matches_df["Explanation"] = "Generated explanation placeholder"  # Add mock explanation column

        # Save the updated CSV
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], "updated_matches.csv")
        matches_df.to_csv(output_path, index=False)
        logging.info(f"Updated matches saved to {output_path}")

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        logging.error(f"Error in process_matches: {e}")
        return jsonify({"error": "Failed to process matches"}), 500

@app.route('/match_students', methods=['POST'])
def match_students():
    try:
        prospective_file = request.files.get("prospective_students_file")
        current_file = request.files.get("current_students_file")

        if not prospective_file or not current_file:
            logging.error("Both files are required for matching.")
            return jsonify({"error": "Both files are required!"}), 400

        prospective_path = os.path.join(app.config['UPLOAD_FOLDER'], "prospective_students.csv")
        current_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_students.csv")

        # Save the uploaded files
        prospective_file.save(prospective_path)
        current_file.save(current_path)
        logging.info("Files successfully uploaded.")

        # Pass file paths to the Celery task
        task = generate_embeddings_task.delay(prospective_path, current_path)
        delete_files.apply_async(args=[[prospective_path, current_path]], countdown=3600)
        logging.info(f"Task ID: {task.id}")

        return render_template("loading.html", task_id=task.id)

    except Exception as e:
        logging.error(f"Error in match_students: {e}")
        return jsonify({"error": "Failed to process student matching"}), 500

@app.route('/task_status/<task_id>')
def task_status(task_id):
    try:
        task = generate_embeddings_task.AsyncResult(task_id)
        logging.info(f"Checking status for Task ID: {task_id}. Current state: {task.state}")

        if task.state == 'SUCCESS':
            csv_path = task.result.get('csv_path')
            if not csv_path:
                logging.error("No CSV path returned from the task result.")
                return jsonify({"status": "FAILURE", "error": "No CSV path in task result"}), 500

            delete_files.apply_async(args=[[csv_path]], countdown=3600)
            filename = os.path.basename(csv_path)
            return jsonify({"status": "SUCCESS", "redirect_url": url_for('results', filename=filename)})

        elif task.state == 'FAILURE':
            error_info = str(task.info)
            logging.error(f"Task failed with error: {error_info}")
            return jsonify({"status": "FAILURE", "error": error_info}), 500

        return jsonify({"status": task.state})

    except Exception as e:
        logging.error(f"Error in task_status: {e}")
        return jsonify({"error": "Failed to retrieve task status"}), 500

@app.route('/results')
def results():
    filename = request.args.get("filename")
    if not filename:
        logging.error("No filename specified in results request.")
        return "No file specified!", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        logging.error(f"Requested file not found: {file_path}")
        return "File not found!", 404

    # Read the results file to extract descriptions
    results_df = pd.read_csv(file_path)
    results_data = results_df.to_dict(orient="records")  # Convert to list of dictionaries

    return render_template("results.html", results_data=results_data, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prospective_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "prospective_students.csv")
        current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_students.csv")

        # Serve the file
        if not os.path.exists(file_path):
            logging.error(f"File not found for download: {file_path}")
            return jsonify({"error": "File not found"}), 404

        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

        # Cleanup the files after download
        for path in [file_path, prospective_file_path, current_file_path]:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Deleted file: {path}")

        return response

    except Exception as e:
        logging.error(f"Error during file download: {e}")
        return jsonify({"error": "Failed to download file"}), 500

if __name__ == "__main__":
    app.run(debug=True)
