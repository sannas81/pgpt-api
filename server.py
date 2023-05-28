from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def process_string():
    data = request.get_json()  # Get the JSON data from the request
    input_string = data['input_string']  # Extract the input string

    print(input_string)
    # Perform any processing or manipulation on the input string here
    output_string = input_string

    # Prepare the response as JSON
    response = {
        'output_string': output_string
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8888)
