# файл нужно положить в папку data, затем
# на маке plistutil может называться plutil
filename="recording-1548683179" # имя файла без .plist
plistutil -i $filename.plist > $filename.xml

# перед следующей командой активировать уже готовый venv: source venv/bin/activate
# ставить пакеты (pip install ...) и создавать venv (virtualenv ...) не нужно
python parse_xml.py $filename
# после всего этого дожны появиться три csv в ./data/csvs/
