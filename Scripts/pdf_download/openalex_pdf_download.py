import os
import json
# import random
import jsonlines
from glob import glob
from tqdm import tqdm
from pprint import pprint
from helper_openalex import download_openalex_pdf, OpenAlexError, InvalidOpenAlexID, PDFNotAvailable, PDFDownloadError

API_KEY = ""
MAX_PAPERS_DOWNLOAD = 814
ROOT_DIR_PATH = "/nfs/home/ahmedf/data/db-arxiv-open"
JSON_DIR_PATH = os.path.join(ROOT_DIR_PATH, "json")
PDF_DIR_PATH = os.path.join(ROOT_DIR_PATH, "pdfs")
INPUT_JSON_FIGURES_ONLY_FILE = os.path.join(ROOT_DIR_PATH, "mappings", "json_with_figures.jsonl")
OUTPUT_JSON_MAPPINT_FILE = os.path.join(ROOT_DIR_PATH, "mappings", "json_pdf_mappings.jsonl")

# Read a whole text file line by line.
json_paths = []
with open(INPUT_JSON_FIGURES_ONLY_FILE, "r") as f:
    json_paths = f.readlines()

print(f"Loaded {len(json_paths)} JSON file paths from {INPUT_JSON_FIGURES_ONLY_FILE}")

download_count = 0
for json_path in tqdm(json_paths):

    try:

        # perform cleanup
        json_path = json_path.strip()

        if download_count >= MAX_PAPERS_DOWNLOAD:
            print(f"Reached maximum download limit of {MAX_PAPERS_DOWNLOAD}. Exiting.")
            exit()

        # Read JSON
        OPEN_ALEX_ID = None
        with open(json_path, 'r') as f:
            json_content = json.load(f)

            # Extract OpenAlex ID
            if "openalex_id" not in json_content:
                print(f"Error: Cannot find openalex_id in JSON file {json_path}.")
            # else:
            OPEN_ALEX_ID = json_content["openalex_id"].split("/")[-1]
            print("OpenAlex ID: ", OPEN_ALEX_ID)

             # if openalex_id exists and PDF file already exists
            pdf_path = os.path.join(PDF_DIR_PATH, f"{OPEN_ALEX_ID}.pdf")
            if OPEN_ALEX_ID and os.path.exists(pdf_path):
                # Saving incremental data
                with jsonlines.open(OUTPUT_JSON_MAPPINT_FILE, mode='a') as writer:
                    writer.write({
                        "pdf_path": str(pdf_path),
                        "json_path": str(json_path)
                    })

                print(f"PDF file for OpenAlex ID {OPEN_ALEX_ID} already exist. Skipping download.")
                continue
          
            # download PDF
            pdf_path = download_openalex_pdf(
                api_key=API_KEY,
                openalex_id=OPEN_ALEX_ID,
                output_folder=PDF_DIR_PATH
            )

            # Saving incremental data
            with jsonlines.open(OUTPUT_JSON_MAPPINT_FILE, mode='a') as writer:
                writer.write({
                    "pdf_path": str(pdf_path),
                    "json_path": str(json_path)
                })
            
            download_count += 1
            
        # with open

    except InvalidOpenAlexID as e:
        # print("The OpenAlex ID does not exist.")
        print(e)

    except PDFNotAvailable as e:
        # print("This paper does not have an Open Access PDF.")
        print(e)

    except PDFDownloadError as e:
        print(e)

    except OpenAlexError as e:
        print(e)

    print("-"*5)

    # for

print(f"Finished downloading {download_count} PDFs.")
