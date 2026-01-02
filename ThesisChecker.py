import fitz  # PyMuPDF
import tiktoken
import ollama
from typing import List
import time
from datetime import timedelta


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrahuje text z PDF souboru.
    """
    print("📖 Čtu PDF soubor...")
    start_time = time.time()

    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n"
        doc.close()

        end_time = time.time()
        print(f"✅ Načteno {len(text)} znaků za {timedelta(seconds=end_time - start_time)}")
        return text

    except Exception as e:
        end_time = time.time()
        print(f"❌ Chyba při čtení PDF ({timedelta(seconds=end_time - start_time)}): {e}")
        return ""


def tokenize_text(text: str, max_tokens: int = 3500, encoding_name: str = "cl100k_base") -> List[str]:
    """
    Rozdělí text na části podle maximálního počtu tokenů.
    Snaží se řezat na koncích vět.
    """
    print("✂️ Rozděluji text na části...")
    start_time = time.time()

    encoding = tiktoken.get_encoding(encoding_name)
    all_tokens = encoding.encode(text)
    total_tokens = len(all_tokens)

    parts = []
    start_index = 0

    while start_index < total_tokens:
        end_index = min(start_index + max_tokens, total_tokens)

        # Pokud nejsme na konci, najdeme hezké místo pro řez
        if end_index < total_tokens:
            current_tokens = all_tokens[start_index:end_index]
            current_text = encoding.decode(current_tokens)

            # Hledáme přirozené konce (tečka, nový řádek)
            last_sentence_end = max(
                current_text.rfind('.'),
                current_text.rfind('!'),
                current_text.rfind('?'),
                current_text.rfind('\n\n'),
                current_text.rfind('\n')
            )

            # Pokud jsme našli dobré místo pro rozdělení
            if last_sentence_end != -1 and last_sentence_end > len(current_text) * 0.6:
                adjusted_text = current_text[:last_sentence_end + 1]
                adjusted_tokens = encoding.encode(adjusted_text)
                end_index = start_index + len(adjusted_tokens)

        # Vytvoříme část textu
        part_tokens = all_tokens[start_index:end_index]
        part_text = encoding.decode(part_tokens)

        parts.append(part_text)
        start_index = end_index

    end_time = time.time()
    print(f"✅ Text rozdělen na {len(parts)} částí za {timedelta(seconds=end_time - start_time)}")
    return parts


def check_grammar_with_ollama(text_chunks: List[str], model_name: str = "jobautomation/OpenEuroLLM-Czech") -> List[str]:
    """
    Pošle části textu modelu ke kontrole.
    """
    print("🚀 Začínám kontrolu gramatiky...")
    overall_start = time.time()
    results = []

    for i, chunk in enumerate(text_chunks):
        chunk_start = time.time()
        print(f"🔍 Kontroluji část {i + 1}/{len(text_chunks)}...", end=" ", flush=True)

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': f"""Zkontroluj tuto část textu z pohledu:
                    1. Gramatiky – vypiš nejčastější chyby v části: [Číslo řádku, Původní text, Oprava, Typ chyby]
                    2. Stylistiky – identifikuj odstavce, které na sebe nenavazují (uvedi čísla odstavců a důvod)
                    3. Odbornosti v oboru ergoterapie – vypiš všechny pasáže, které jsou odborně nesprávné, s vysvětlením.

                    Text: {chunk}

                    Poznámka: Ignoruj abstrakt, seznam literatury, přílohy, obsah, seznam obrázků a tabulek.
                    Odpověď dej stručně, v češtině.
                    
                    Na úplný záver tvého hodnocení přídej celkové shrnutí s procentuálním vyjádřením kvality textu z hlediska gramatiky, stylistiky a odbornosti (0-100%). Buď konkrétní a věcný a mužeš zahrnout i návrhy na zlepšení.
                    
                    """
                }]
            )
            results.append(response['message']['content'])

            chunk_end = time.time()
            chunk_time = chunk_end - chunk_start
            print(f"hotovo za {chunk_time:.1f}s")

        except Exception as e:
            chunk_end = time.time()
            chunk_time = chunk_end - chunk_start
            print(f"❌ Chyba za {chunk_time:.1f}s: {e}")
            results.append(f"CHYBA: {e}")

    overall_end = time.time()
    total_time = overall_end - overall_start
    avg_time = total_time / len(text_chunks) if text_chunks else 0

    print(f"✅ Kontrola dokončena za {timedelta(seconds=total_time)}")
    print(f"📊 Průměrně {avg_time:.1f}s na část")

    return results


def main(pdf_path: str):
    """
    Hlavní funkce: PDF → Text → Tokenizace → Kontrola
    """
    print("=" * 60)
    print("🤖 SPUŠTĚNÍ KONTROLY GRAMATIKY")
    print("=" * 60)

    total_start_time = time.time()

    # Fáze 1: Extrakce textu z PDF
    text = extract_text_from_pdf(pdf_path)

    if not text:
        print("❌ Nepodařilo se načíst text z PDF")
        return

    # Fáze 2: Tokenizace
    chunks = tokenize_text(text, max_tokens=3000)

    # Fáze 3: Kontrola gramatiky
    results = check_grammar_with_ollama(chunks)

    # Fáze 4: Uložení výsledků
    print("💾 Ukládám výsledky...")
    save_start = time.time()

    with open("vysledky_kontroly.txt", "w", encoding="utf-8") as f:
        for i, result in enumerate(results):
            f.write(f"\n{'=' * 50}\n")
            f.write(f"ČÁST {i + 1}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(result + "\n")

    save_end = time.time()

    # Celkové statistiky
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print("=" * 60)
    print("📊 CELKOVÉ STATISTIKY")
    print("=" * 60)
    print(f"Celkový čas: {timedelta(seconds=total_duration)}")
    print(f"Počet částí: {len(chunks)}")
    print(f"Průměrný čas na část: {total_duration / len(chunks):.1f}s" if chunks else "N/A")
    print(f"Čas uložení: {timedelta(seconds=save_end - save_start)}")
    print(f"✅ Hotovo! Výsledky uloženy v 'vysledky_kontroly.txt'")
    print("=" * 60)


# Spuštění
if __name__ == "__main__":
    #main("130416806.pdf")  # 👈 ZDE NAZEV TVÉHO SOUBORU
    main("Lopatka.pdf")  # 👈 ZDE NAZEV TVÉHO SOUBORU