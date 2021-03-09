import predict

def main():
    conversation = []
    while True:
        sentence = input("Input: ")
        conversation.append(sentence)
        while len(conversation) > predict.CONVERSATION_TURNS: 
            conversation.pop(0)
        response, score = predict.predict(conversation)
        conversation.append(response)
        print(response)

if __name__ == '__main__':
  main()