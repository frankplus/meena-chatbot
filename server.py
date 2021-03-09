from flask import Flask, request
import secrets
import string
import predict

app = Flask(__name__)
enabled_api_keys = ["f24eded9-fcd1-4392-b214-01bad08fa69f"]
contexts = dict()


def generate_context_id():
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(64))


@app.route('/getreply', methods=['GET'])
def getreply():
    key = request.args.get('key')
    if not key or key not in enabled_api_keys:
        return {
            "status": "401",
            "error": "Missing or invalid API key"
        }, 401

    text = request.args.get('input')
    if text:
        print(f"Input: {text}")

    context_id = request.args.get('context')
    if not context_id:
        context_id = generate_context_id()

    if context_id not in contexts:
        contexts[context_id] = list()
        
    contexts[context_id].extend(text.split('\n'))
    while len(contexts[context_id]) > predict.CONVERSATION_TURNS: 
        contexts[context_id].pop(0)

    # elaborate response
    answer, score = predict.predict(contexts[context_id])
    print(f"Score: {score} Answer: {answer}")
    contexts[context_id].append(answer)

    return {
        "context": context_id,
        "output": str(answer),
        "score": score
    }


if __name__ == '__main__':
    app.run("localhost", "2834", debug=True)
