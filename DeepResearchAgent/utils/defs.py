from bs4 import BeautifulSoup
import re
import psycopg2
import os

def editDbP(s_query, a_values):
    i_return = -1
    try:
        conn = psycopg2.connect(database="xxxx",
        host="xxxx",
        user="xxxx",
        password="xxxx",
        port="5432")
        cursor = conn.cursor()

        cursor.execute(s_query, a_values)

        conn.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into table")

        i_return = cursor.fetchone()[0]

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into table", error)
    # i_return = -2

    finally:
        # closing database connection.
        if conn:
            cursor.close()
            conn.close()
            print("PostgreSQL connection is closed")

    return i_return

def clean_text(text):
    # Remove extra whitespace while preserving single spaces between words
    return ' '.join(text.split())

def extract_images_and_captions(html_content):
    # Create BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # List to store image data
    images_data = []
    
    # Find all images
    images = soup.find_all('img')
    
    for img in images:
        # Get image source
        src = img.get('src')
        if src and src.startswith('//'):
            src = 'https:' + src
        
        # Initialize caption
        caption = None
        
        # Look for caption in various parent elements
        parent = img.parent
        while parent and not caption:
            # Check next sibling for caption if parent is a table cell
            if parent.name == 'td':
                next_row = parent.find_parent('tr').find_next_sibling('tr')
                if next_row:
                    caption = clean_text(next_row.get_text())
            
            # Check if there's a figcaption in the parent
            figcaption = parent.find('figcaption')
            if figcaption:
                caption = clean_text(figcaption.get_text())
            
            # Move up to next parent
            parent = parent.parent
        
        # Add to list if we have both src and caption
        if src and caption:
            images_data.append({
                'src': src,
                'caption': caption
            })
    
    return images_data
