import re

def clean_section_text(text):
    """
    Remove the article header up to the next uppercase letter.
    """
    # Pattern to match from article header to next uppercase letter
    header_pattern = r'^(?:ARTÍCULO|ARTICULO|Artículo|Articulo)\s+\d+.*?(?=[A-ZÁÉÍÓÚÑ])'
    cleaned_text = re.sub(header_pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_article_sections(text):
    """
    Extract sections of text between article markers and assign article numbers.
    Only matches when starting with capital A ('ARTÍCULO' or 'Artículo').
    Returns a list of (text, article_number) tuples with cleaned section text.
    """
    # Pattern that only matches when starting with capital A
    article_pattern = r'(?:ARTÍCULO|ARTICULO|Artículo|Articulo)\s+(\d+)'
    
    # Find all article markers with their positions
    article_matches = list(re.finditer(article_pattern, text))
    
    article_sections = []
    
    # Process each article section
    for i in range(len(article_matches)):
        start_pos = article_matches[i].start()
        article_num = article_matches[i].group(1)

        # Get end position (either next article or end of text)
        if i < len(article_matches) - 1:
            end_pos = article_matches[i + 1].start()
        else:
            end_pos = len(text)
        
        section_text = text[start_pos:end_pos]
        # Clean the section text by removing the header
        cleaned_section = clean_section_text(section_text)
        article_sections.append((cleaned_section, article_num))
    
    return article_sections