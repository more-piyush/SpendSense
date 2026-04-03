import zipfile
import xml.etree.ElementTree as ET

docx_path = r'C:\Users\USER\Desktop\Academics\MLOPs\Project\Documentation\Firefly_III_Complete_Project_Documentation.docx'

with zipfile.ZipFile(docx_path, 'r') as z:
    xml_content = z.read('word/document.xml')

tree = ET.fromstring(xml_content)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

paragraphs = tree.findall('.//w:p', ns)
full_text = []
for p in paragraphs:
    texts = p.findall('.//w:t', ns)
    para_text = ''.join(t.text or '' for t in texts)
    full_text.append(para_text)

print('\n'.join(full_text))
