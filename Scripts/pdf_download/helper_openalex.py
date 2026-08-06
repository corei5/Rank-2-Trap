from pathlib import Path
import requests


# ----------------------------------------------------------------------
# Custom Exceptions
# ----------------------------------------------------------------------

class OpenAlexError(Exception):
    """Base exception for OpenAlex download errors."""
    pass


class InvalidOpenAlexID(OpenAlexError):
    """Raised when the OpenAlex ID does not exist."""
    pass


class PDFNotAvailable(OpenAlexError):
    """Raised when no downloadable PDF is available."""
    pass


class PDFDownloadError(OpenAlexError):
    """Raised when the PDF download fails."""
    pass


# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def download_openalex_pdf(
    api_key: str,
    openalex_id: str,
    output_folder: str,
    timeout: int = 30,
) -> Path:
    """
    Download an Open Access PDF for an OpenAlex work.

    Parameters
    ----------
    openalex_id : str
        OpenAlex work ID.
        Examples:
            "W2741809807"
            "https://openalex.org/W2741809807"

    output_folder : str
        Directory where the PDF will be saved.

    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pathlib.Path
        Path to the downloaded PDF.

    Raises
    ------
    InvalidOpenAlexID
        If the OpenAlex work does not exist.

    PDFNotAvailable
        If no OA PDF is available.

    PDFDownloadError
        If downloading the PDF fails.

    OpenAlexError
        For other API-related errors.
    """

    # ------------------------------------------------------------
    # Normalize ID
    # ------------------------------------------------------------
    if openalex_id.startswith("http"):
        work_id = openalex_id.rstrip("/").split("/")[-1]
    else:
        work_id = openalex_id

    # api_url = f"https://api.openalex.org/works/{work_id}"
    api_url = f"https://api.openalex.org/works/{work_id}?api_key={api_key}"
    # api_url = f"https://content.openalex.org/works/{work_id}.pdf?api_key={api_key}"

    # ------------------------------------------------------------
    # Query OpenAlex
    # ------------------------------------------------------------
    try:
        response = requests.get(api_url, timeout=timeout)
    except requests.RequestException as e:
        raise OpenAlexError(f"Unable to connect to OpenAlex: {e}") from e

    if response.status_code == 404:
        raise InvalidOpenAlexID(f"OpenAlex work '{work_id}' was not found.")

    if response.status_code != 200:
        raise OpenAlexError(
            f"OpenAlex API returned HTTP {response.status_code}."
        )

    work = response.json()

    # ------------------------------------------------------------
    # Find PDF URL
    # ------------------------------------------------------------
    pdf_url = None

    # Preferred location
    oa_location = work.get("best_oa_location")
    if oa_location:
        pdf_url = oa_location.get("pdf_url")

    # Fallback
    if not pdf_url:
        primary = work.get("primary_location")
        if primary:
            pdf_url = primary.get("pdf_url")

    if not pdf_url:
        raise PDFNotAvailable(
            f"No Open Access PDF available for '{work_id}'."
        )

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{work_id}.pdf"

    # ------------------------------------------------------------
    # Download PDF
    # ------------------------------------------------------------
    try:
        with requests.get(pdf_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type and "application/octet-stream" not in content_type:
                raise PDFDownloadError(
                    f"URL did not return a PDF "
                    f"(Content-Type={content_type})."
                )

            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    except requests.RequestException as e:
        raise PDFDownloadError(f"Failed to download PDF: {e}") from e

    if output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise PDFDownloadError("Downloaded PDF is empty.")

    return output_path

