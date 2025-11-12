
# ocr.py

from __future__ import annotations
import re
from datetime import date
from io import BytesIO
from typing import Dict, Optional, Any

# Optional import for OCR
try:
    from PIL import Image
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
except Exception:
    # Tesseract might be installed but not in PATH
    HAS_TESSERACT = False

# Fallback for Tesseract executable (adjust as needed for local environment)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example path


def extract_bill_fields(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Extracts transaction fields (amount, date, merchant) from an image of a bill/receipt.
    
    Args:
        image_bytes: The image content as bytes (e.g., from st.file_uploader).
        
    Returns:
        Dict|None: Extracted fields, or None on failure.
    """
    if not HAS_TESSERACT:
        return None

    try:
        img = Image.open(BytesIO(image_bytes))
        # Use English + Hindi for robustness
        text = pytesseract.image_to_string(img, lang='eng+hin') 
        
        # --- Extraction Logic ---
        
        # 1. Amount: Look for "Total", "Amount", "Balance" near an INR value
        # Pattern: (₹|INR|RS|Rs|Total).*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)
        amount_match = re.search(r'(?:TOTAL|AMOUNT|BALANCE|NET)\s*[:\-\s]*\s*(?:₹|INR|RS\s*|Rs\s*)\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)', text, re.IGNORECASE)
        amount_val = None
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '').replace(' ', '')
            try:
                amount_val = float(amount_str)
            except ValueError:
                pass

        # Fallback: Look for large numbers that might be a total
        if amount_val is None:
             amount_match_2 = re.findall(r'\d{3,}(?:\.\d{2})?', text)
             if amount_match_2:
                 # Assume the largest is the total
                 amount_val = max([float(a) for a in amount_match_2 if float(a) > 50], default=None)
        
        # 2. Date: Look for common date formats
        date_match = re.search(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text)
        date_val = date.today()
        if date_match:
            try:
                # Ensure date parsing is robust
                date_str = date_match.group(1)
                date_val = pd.to_datetime(date_str, errors='coerce', dayfirst=True).date()
            except Exception:
                pass
        
        # 3. Merchant: Take the first line (common for receipt header)
        merchant_name = text.split('\n')[0].strip() if text else "Scanned Merchant"
        
        if amount_val is None:
            return None # Must have an amount to be useful

        return {
            "amount": amount_val,
            "date": date_val.isoformat(),
            "merchant": merchant_name,
            "raw_text": text[:500] + "..." # Include some raw text for debugging
        }
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return None