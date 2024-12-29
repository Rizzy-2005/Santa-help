from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from flask_cors import CORS
import os
from dotenv import load_dotenv
import replicate
import traceback


load_dotenv()

app = Flask(__name__)
CORS(app)


token = os.getenv('REPLICATE_API_TOKEN')
if not token:
    raise ValueError("REPLICATE_API_TOKEN not found in environment variables!")
else:
    print("Replicate token loaded successfully!")

@app.route("/generate_reply", methods=["POST"])
def generate_reply():
    try:
        
        print("Received request data:", request.get_json())
        
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        name = data.get("name")
        age = data.get("age")
        letter = data.get("letter")
        language = data.get("language", "en")
        
     
        if not all([name, age, letter]):
            return jsonify({"error": "Missing required fields"}), 400
            
        print(f"Processing letter from {name}, age {age}, in {language}")
        
        
        try:
            if language != "en":
                translated_to_english = GoogleTranslator(source="auto", target="en").translate(letter)
                print(f"Translated text: {translated_to_english}")
            else:
                translated_to_english = letter
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return jsonify({"error": "Translation failed"}), 500

       
        try:
            prompt = f"""You are Santa Claus responding to a child's letter. Please write a warm, kind, and festive reply 
            to this {age}-year-old child named {name} who wrote:
            '{translated_to_english}'
            
            Write a response that is about 100 words long and maintains the magic of Christmas.
            """
            
            
            output = replicate.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={
                    "prompt": prompt,
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "system_prompt": "You are Santa Claus, responding to children's letters with warmth and holiday cheer."
                }
            )
            
            
            santa_reply = "".join(output).strip()
            print(f"Generated Santa's reply: {santa_reply}")
            
        except Exception as e:
            print(f"Llama API error: {str(e)}")
            return jsonify({"error": "Failed to generate Santa's reply"}), 500

        
        try:
            if language != "en":
                translated_reply = GoogleTranslator(source="en", target=language).translate(santa_reply)
                print(f"Translated reply: {translated_reply}")
            else:
                translated_reply = santa_reply
        except Exception as e:
            print(f"Reply translation error: {str(e)}")
            return jsonify({"error": "Failed to translate Santa's reply"}), 500

        return jsonify({
            "original_letter": letter,
            "translated_letter": translated_to_english,
            "santa_reply": translated_reply
        })

    except Exception as e:
        print("ERROR TRACEBACK:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   
    try:
        test_output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "prompt": "Say 'Hello'",
                "max_new_tokens": 10,
            }
        )
        print("Successfully connected to Replicate!")
    except Exception as e:
        print(f"Warning: Could not connect to Replicate: {str(e)}")
    
    app.run(debug=True)