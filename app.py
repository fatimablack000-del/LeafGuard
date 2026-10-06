import os
import json
import requests

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    static_folder=".",
    static_url_path=""
)


# =========================================================
# GEMINI SETTINGS
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/interactions"
)

GEMINI_MODEL = "gemini-3.8-flash"


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():
    print("FLASK CURRENT FOLDER:", os.getcwd())
    print("FLASK INDEX FILE:", os.path.abspath("index.html"))
    return send_from_directory(".", "index.html")


# =========================================================
# GEMINI PLANT ANALYSIS
# =========================================================

@app.route("/api/analyze-plant", methods=["POST"])
def analyze_plant():

    try:

        # -------------------------------------------------
        # Check API key
        # -------------------------------------------------

        if not GEMINI_API_KEY:

            return jsonify({
                "error": (
                    "Gemini API key is not configured. "
                    "Please add GEMINI_API_KEY to your .env file."
                )
            }), 500


        # -------------------------------------------------
        # Read JSON sent by the existing HTML
        # -------------------------------------------------

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "error": "No analysis data was received."
            }), 400


        image_base64 = data.get("image")
        mime_type = data.get(
            "mimeType",
            "image/jpeg"
        )
        nickname = data.get(
            "nickname",
            ""
        ).strip()


        # -------------------------------------------------
        # Validate image
        # -------------------------------------------------

        if not image_base64:

            return jsonify({
                "error": "No plant image was provided."
            }), 400


        # -------------------------------------------------
        # Plant context
        # -------------------------------------------------

        if nickname:

            plant_context = (
                f"The user identified the plant as: {nickname}. "
                "Use this only as supporting information. "
                "Do not assume it is correct if the image "
                "suggests otherwise."
            )

        else:

            plant_context = (
                "The user did not provide a plant name. "
                "Identify the plant only if the image provides "
                "enough visual evidence."
            )


        # =================================================
        # GEMINI PROMPT
        # =================================================

        prompt = f"""
You are LeafGuard, an AI-assisted plant health analysis system.

Analyze the provided image of a plant or leaf.

{plant_context}

Assess the visible condition of the plant.

Look for:

- pests
- diseases
- fungal symptoms
- leaf discoloration
- leaf spots
- yellowing
- wilting
- physical damage
- overwatering symptoms
- underwatering symptoms
- nutrient-related symptoms
- other visible plant stress

IMPORTANT:

- Base your assessment primarily on what is visible.
- Do not invent symptoms.
- Do not claim certainty when the image is unclear.
- If the image does not provide enough evidence, state that
  the result is uncertain.
- This is an informational plant-care tool and not a
  professional agricultural diagnosis.

Return ONLY a JSON object using exactly these fields:

plantName
confidence
healthIssue
symptoms
causes
recommendations
careSpecs

The fields must follow this structure:

{{
    "plantName": "string",
    "confidence": 0,
    "healthIssue": "string",
    "symptoms": [
        "string"
    ],
    "causes": "string",
    "recommendations": [
        "string"
    ],
    "careSpecs": {{
        "water": "string",
        "sun": "string",
        "soil": "string"
    }}
}}

For confidence, return a number from 0 to 100.

The causes field should be a concise explanation
of the possible cause or causes.

Provide practical and safe recommendations.
"""


        # =================================================
        # GEMINI REQUEST
        # =================================================

        payload = {

            "model": GEMINI_MODEL,

            "input": [

                {
                    "type": "text",
                    "text": prompt
                },

                {
                    "type": "image",
                    "data": image_base64,
                    "mime_type": mime_type
                }

            ],

            "response_format": {

                "type": "text",

                "mime_type": "application/json",

                "schema": {

                    "type": "object",

                    "properties": {

                        "plantName": {
                            "type": "string"
                        },

                        "confidence": {
                            "type": "integer"
                        },

                        "healthIssue": {
                            "type": "string"
                        },

                        "symptoms": {

                            "type": "array",

                            "items": {
                                "type": "string"
                            }

                        },

                        "causes": {
                            "type": "string"
                        },

                        "recommendations": {

                            "type": "array",

                            "items": {
                                "type": "string"
                            }

                        },

                        "careSpecs": {

                            "type": "object",

                            "properties": {

                                "water": {
                                    "type": "string"
                                },

                                "sun": {
                                    "type": "string"
                                },

                                "soil": {
                                    "type": "string"
                                }

                            },

                            "required": [
                                "water",
                                "sun",
                                "soil"
                            ]

                        }

                    },

                    "required": [
                        "plantName",
                        "confidence",
                        "healthIssue",
                        "symptoms",
                        "causes",
                        "recommendations",
                        "careSpecs"
                    ]

                }

            }

        }


        # =================================================
        # SEND REQUEST TO GEMINI
        # =================================================

        headers = {

            "x-goog-api-key":
                GEMINI_API_KEY,

            "Content-Type":
                "application/json"

        }


        response = requests.post(

            GEMINI_URL,

            headers=headers,

            json=payload,

            timeout=120

        )


        # =================================================
        # HANDLE GEMINI ERRORS
        # =================================================

        if not response.ok:

            try:

                error_details = response.json()

            except Exception:

                error_details = response.text


            print(
                "Gemini API error:",
                error_details
            )


            return jsonify({

                "error":
                    "Gemini API request failed.",

                "details":
                    error_details

            }), response.status_code


        # =================================================
        # READ GEMINI RESPONSE
        # =================================================

        gemini_data = response.json()


        output_text = (
            gemini_data.get("output_text")
        )


        # -------------------------------------------------
        # Fallback response extraction
        # -------------------------------------------------

        if not output_text:

            for step in gemini_data.get(
                "steps",
                []
            ):

                if step.get(
                    "type"
                ) == "model_output":

                    for content in step.get(
                        "content",
                        []
                    ):

                        if content.get(
                            "type"
                        ) == "text":

                            output_text = (
                                content.get(
                                    "text"
                                )
                            )

                            break


                if output_text:
                    break


        # -------------------------------------------------
        # Check output
        # -------------------------------------------------

        if not output_text:

            print(
                "Gemini response:",
                gemini_data
            )

            return jsonify({

                "error":
                    "Gemini returned no analysis."

            }), 500


        # =================================================
        # CONVERT GEMINI JSON
        # =================================================

        try:

            analysis = json.loads(
                output_text
            )

        except json.JSONDecodeError:

            print(
                "Invalid Gemini JSON:",
                output_text
            )

            return jsonify({

                "error":
                    "Gemini returned an invalid analysis format."

            }), 500


        # =================================================
        # RETURN RESULT TO YOUR EXISTING HTML
        # =================================================

        return jsonify({

            "success": True,

            "analysis": analysis,

            "model": GEMINI_MODEL

        })


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except requests.Timeout:

        return jsonify({

            "error":
                "Gemini took too long to respond. "
                "Please try again."

        }), 504


    except requests.RequestException as e:

        print(
            "Request error:",
            str(e)
        )

        return jsonify({

            "error":
                "Could not connect to Gemini."

        }), 502


    except Exception as e:

        print(
            "Server error:",
            str(e)
        )

        return jsonify({

            "error":
                "An unexpected server error occurred.",

            "details":
                str(e)

        }), 500


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )