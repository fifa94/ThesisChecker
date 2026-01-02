import ollama

def ohodnot_gramatiku(text):
    """
    Pošle text modelu a požádá ho o hodnocení gramatické úrovně.
    """
    
    # Připravíme si prompt, který model navede k hodnocení
    system_prompt = """Jsi expertní linguista a hodnotitel českého jazyka. 
    Tvým úkolem je ohodnit gramatickou a stylistickou úroveň akademického textu.
    
    Proveď kompletní analýzu textu a:
    1. Vypiš celkové skóre na škále 1-10 (kde 10 je bezchybný akademický text)
    2. Vypiš počet nalezených chyb (pravopisných, gramatických, stylistických)
    3. Vypiš 3-5 nejzávažnějších chyb s konkrétními příklady a návrhy oprav
    4. Uveď celkové hodnocení úrovně textu
    
    Odpověz formátovaně v češtině."""

    try:
        # Odeslání požadavku modelu
        response = ollama.chat(
            model='open-euro-llm-czech',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Text k hodnocení:\n\n{text}"}
            ]
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"Chyba při komunikaci s modelem: {str(e)}"

def main():
    """
    Hlavní funkce programu
    """
    print("🤖 Hodnotitel gramatické úrovně textu")
    print("=" * 50)
    
    # Můžeš buď načíst text ze souboru...
    # with open('text_k_hodnoceni.txt', 'r', encoding='utf-8') as f:
    #     text = f.read()
    
    # ...nebo vložit text přímo zde
    text = """
    V této práci se budu zabývat analýzou dat. Data jsem sbíral během 
    letních měsícu. Výsledky jsou vidět v grafu číslo dvě. Myslím si že 
    se hypotéza potvrdila což je dobry. V budoucnu by se to dalo dělat 
    jinak a lepší.
    """
    
    print("📝 Analyzovaný text:")
    print("-" * 30)
    print(text)
    print("-" * 30)
    
    # Získání hodnocení od modelu
    print("\n🔍 Analyzuji text...", end="", flush=True)
    hodnoceni = ohodnot_gramatiku(text)
    
    print(" hotovo!\n")
    
    # Výpis výsledku
    print("📊 VÝSLEDEK HODNOCENÍ:")
    print("=" * 50)
    print(hodnoceni)

if __name__ == "__main__":
    main()