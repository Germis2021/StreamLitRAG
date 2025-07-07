# StreamLitRAG


# Klaipėda Chatbotas su LangChain ir OpenAI

Ši Streamlit aplikacija sukurta naudojant LangChain ir OpenAI GPT modelį. Ji leidžia užduoti klausimus apie Klaipėdą, o chatbotas atsako remdamasis įvairiais duomenų šaltiniais – interneto puslapiais, vietiniu tekstiniu failu ir PDF dokumentu.

---

## Funkcijos

- Automatiškai įkeliami ir apdorojami keli dokumentų šaltiniai (web, TXT, PDF).
- Dokumentai suskaidomi į tekstinius fragmentus (chunk'us).
- Naudoja OpenAI embedding'us ir GPT-4.1 modelį atsakymams generuoti.
- Rodo naudotus informacijos fragmentus kartu su šaltinių aprašymais Streamlit aplikacijoje.
- Vartotojui patogi pokalbio sąsaja.

---

## Reikalingos bibliotekos

```bash
pip install streamlit python-dotenv langchain-openai langchain langchain-community pypdf





Paruošimas
Sukurk .env failą projekto šaknyje su savo OpenAI API raktu:

ini
Kopijuoti
Redaguoti
SECRET= tavo_openai_api_raktas
Įkelk šiuos failus į projekto aplanką:

Klaipeda.txt (vietinis tekstinis failas su informacija apie Klaipėdą)

Klaipeda-Wikipedia.pdf (PDF dokumentas)

Jei reikia, pakeisk PDF failo kelią kode (kintamasis pdf_path).

Paleidimas
bash
Kopijuoti
Redaguoti
streamlit run tavo_failo_pavadinimas.py
Atidarys naršyklėje Streamlit aplikaciją, kur galėsi užduoti klausimus apie Klaipėdą ir matyti atsakymus kartu su naudotais šaltiniais.

Kodo struktūra
Dokumentų įkėlimas (web, TXT, PDF)

Dokumentų suskaidymas į chunk'us

Embedding'ų kūrimas ir vektorinės paieškos įdiegimas

RAG (retriever-augmented generation) grandinė su OpenAI GPT

Streamlit UI su pokalbio istorija ir naudotų šaltinių atvaizdavimu

Kontaktai
Jei kyla klausimų ar pasiūlymų, kreipkitės: https://github.com/Germis2021
