Welkom bij de Rooster automatiseringstool!
Deze tool is ontworpen om Het Gastenhuis te helpen om hun roosterbeheer te automatiseren en hier efficiënter mee om te gaan.
De tool is opgebouwd met verschillende modules:
1 **app**: In deze module bevindt zich alle kernfunctionaliteiten van de tool, inclusief de logica voor het maken van de roosters. In deze module zijn 3 verschillende python bestanden te vinden:
   - **preprocessing.py**: Dit bestand bevat functies voor het voorbereiden van de data voordat deze wordt gebruikt voor het genereren van roosters.
   - **solver.py**: Dit bestand bevat de logica voor het oplossen van het roosterprobleem en het genereren van de roosters. 
   - **validate.py**: Dit bestand bevat functies voor het valideren van de gegenereerde roosters om ervoor te zorgen dat ze voldoen aan de gestelde eisen.
2 **web**: Deze module bevat de webinterface van de tool die gemaakt zijn in Flask, waarmee gebruikers kunnen communiceren met de applicatie. In deze module zijn verschillende bestanden te vinden die verantwoordelijk zijn voor het weergeven van de webpagina's en het afhandelen van gebruikersinteracties:
   - **templates/**: Deze map bevat HTML-bestanden die de structuur en inhoud van de webpagina's definiëren.
   - **static/**: Deze map bevat statische bestanden zoals afbeeldingen die worden gebruikt in de webinterface.
   - **app.py**: Dit bestand bevat de route-definities voor de webapplicatie, waarmee wordt bepaald welke functies worden aangeroepen voor specifieke URL's.

Om de tool te gebruiken, wordt Docker gebruikt om een consistente omgeving te garanderen. Volg de onderstaande stappen om de tool op te zetten en uit te voeren:
1. Zorg ervoor dat Docker op uw systeem is geïnstalleerd. U kunt Docker downloaden en installeren vanaf de officiële website: https://www.docker.com/get-started
2. Open Docker Desktop en open de ingebouwde terminal. De terminal is aan de rechteronderkant van het scherm te vinden.
![alt text](https://github.com/Jagovanlieshout/Gastenhuis_Autorooster/blob/main/Images/Readme_1.png)
3. In de terminal moet je de volgende commands runnen:
git clone https://github.com/Jagovanlieshout/Gastenhuis_Autorooster.git
cd Hoevelaken_autorooster
docker build -t rooster_automaat .
4. Klik in het menu aan de linkerkant op 'Images' hier zie je nu een image genaamd rooster_automaat staan, dit is wat we zojuist hebben opgezet.
5. Klik op de start knop aan de rechter kant. Open daarna de 'Optional settings'.
6. Verander de naam naar rooster_automaat en vul 8000 in bij port. Klik hierna op run.
7. Open je webbrowser en ga naar http://localhost:8000 om de webinterface van de tool te openen.
Veel succes met het gebruik van de Rooster automatiseringstool! 
