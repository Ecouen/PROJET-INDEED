# IMPORTANT: les commentaires todo sont des instructions à suivre lors du Pre Processing (PP)

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import re
import pandas as pd
import sys
import datetime

driver = webdriver.Firefox(executable_path='./geckodriver.exe') #REMPLACER LE DRIVER PAR CHROME OU AUTRE SI NECESSAIRE
driver.get("https://www.indeed.fr/")

seconds = 1  # Temps d'attente du chargement de la page avant scraping
is_last_page = False  # Est ce qu on se trouve sur la dernière page

############### Valeurs à modif pour paramétrer le scraping ###############
mot_cle_job = "Data Scientist Senior"  # Job recherché
mot_cle_localisation = "Paris"  # Localisation
sort_by_date = True # Si false: les résultats sont affichées par pertnience
delai_scraping_entre_chaque_resultats = 0.1 # si le pc charge la page rapidement, ne pas hésiter à baisser cette valeur, sinon augmenter pour réduire le risque d'interruption du scraping (exceptions)
###########################################################################

# creation de listes temporaires pour stocker les informations scrapées. Ces listes seront par la suite transformées en dataframe puis en CSV
id = []
meta1 = []
meta2 = []
titre_poste = []
entreprise = []
localisation = []
contrat = []
salaire = []
date_publication = []
description = []
keyword_metier = []
keyword_localisation = []

# actions xpath
def recherche_xpath(str_xpath, action, str_recherche="", delay=0):
    if (delay > 0):
        time.sleep(delay)

    if (action == 0):  # sélection de l'élément
        return driver.find_element_by_xpath(str_xpath)
    elif (action == 1):  # sélection + clique
        return driver.find_element_by_xpath(str_xpath).click()
    elif (action == 2):  # sélection + clique + recherche
        driver.find_element_by_xpath(str_xpath).click()
        driver.find_element_by_xpath(str_xpath).send_keys(str_recherche)
    elif (action == 3):  # sélection + clique + saisie de la recherche + effectuer recherche
        driver.find_element_by_xpath(str_xpath).click()

        # pour être sûr que la case localisation est vide avant que l'on y entre notre localisation souhaitée
        driver.find_element_by_xpath(str_xpath).send_keys(Keys.CONTROL + 'a') # windows
        #driver.find_element_by_xpath(str_xpath).send_keys(Keys.COMMAND + 'a') # macos

        driver.find_element_by_xpath(str_xpath).send_keys(Keys.DELETE)

        driver.find_element_by_xpath(str_xpath).send_keys(str_recherche)
        time.sleep(seconds)
        return driver.find_element_by_xpath(str_xpath).send_keys((Keys.ENTER))

# scraping page
def scrap_current_page():
    time.sleep(seconds)  # On attend que la page se charge complètement

    # S'il y a un popup, on le ferme
    try:
        driver.find_element_by_xpath('//*[@id="popover-x"]').click()  # On ferme le popup
    except:  # block à exécuter si le try ne fonctionne pas
        pass

    resultats_emplois = driver.find_element_by_id('resultsCol')
    liste_emplois = resultats_emplois.find_elements_by_class_name('title')

    # Scraping des résultats de la page affichée
    for i in range(len(liste_emplois)):
        #print("********************************************************************* NOUVEAU RESULTAT ***************************************************************************")

        liste_emplois[i].click()
        time.sleep(delai_scraping_entre_chaque_resultats)

        #print("----------------------URL-----------------------")
        id.append(driver.current_url)

        #print("----------------------INFOS META 1-----------------------")
        try:
            meta1.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-header-jobinfo"]').text)
        except:
            meta1.append("null")
            pass

        #print("----------------------INFOS META 2-----------------------")
        tmp_meta2 = []
        try:
            tmp_meta2.append(liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[1]/span[2]').text)
            tmp_meta2.append(liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[2]/span[2]').text)
            tmp_meta2.append(liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[3]/span[2]').text)
            meta2.append(tmp_meta2)
        except:
            try:
                meta2.append(tmp_meta2)
            except:
                meta2.append("NULL")
                pass
        #todo: scraping du meta, au cas où des informations manqueraient sur le salaire/type de contrat/temps plein etc.

        # print("----------------------TITRE POSTE 1-----------------------")
        try:
            #print(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-jobtitle"]').text)
            titre_poste.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-jobtitle"]').text)
        except:
            # print("----------------------TITRE POSTE 2-----------------------")
            try:
                #print(liste_emplois[i].find_element_by_class_name('jobtitle turnstileLink visited').text)
                titre_poste.append(liste_emplois[i].find_element_by_class_name('jobtitle turnstileLink visited').text)
            except:
                titre_poste.append("null")
                pass
        # todo: si le titre est manquant, faire un regex/NLP depuis la description du poste

        #print("----------------------NOM DE L'ENTREPRISE-----------------------")
        #print(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-cn"]').text)
        try:
            entreprise.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-cn"]').text)
        except:
            entreprise.append("null")
            pass

        # print("----------------------LOCALISATION-----------------------")
        try:
            #print(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-loc"]').text)
            localisation.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-loc"]').text)
        except:
            localisation.append("null")
            pass
        # todo: si elle n'est pas affichée, on utilise le mot clé de localisation pour combler

        # print("----------------------TYPE DE CONTRAT-----------------------")
        try:
            #print(liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[2]').text)
            contrat.append(liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[2]').text)
        except:
            contrat.append("null")
            pass
        # todo: effectuer la séparation "type de contrat"/"temps plein ou partiel"
        # todo: si valeur manquante, essayer un regex/NLP sur la description pour avoir l'information

        # print("----------------------SALAIRE-----------------------")
        try:
            str_salaire = liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[2]/span[2]').text

            if '€' in str_salaire and len(str_salaire) < 150:
                    salaire.append(str_salaire)
            else:
                str_salaire = liste_emplois[i].find_element_by_xpath('/html/body/table[2]/tbody/tr/td/table/tbody/tr/td[3]/div/div[2]/div[1]/div[1]/div[3]/span[2]').text
                if '€' in str_salaire and len(str_salaire) < 150:
                    salaire.append(str_salaire)
                else:
                    salaire.append("null")
        except:
            salaire.append("null")
            pass
        #todo: REGEX/NLP salaire pour avoir des int

        # print("----------------------DESCRIPTION-----------------------")
        try:
            #print(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-desc"]').text)
            description.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-desc"]').text)
        except:
            description.append("null")
            pass
        #todo: extraire les infos depuis cette description
        #todo: attention aux postes en anglais

        # print("----------------------DATE DE PUBLICATION DE L'ANNONCE----------------------")
        try:
            #print(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-footer"]').text)
            date_publication.append(liste_emplois[i].find_element_by_xpath('//*[@id="vjs-footer"]').text)
        except:
            date_publication.append("null")
            pass
        #todo: trouver un moyen de transformer cette valeur en datetime

        #print("----------------------KEYWORDS UTILISES POUR LA RECHERCHE----------------------")
        keyword_localisation.append(mot_cle_localisation)
        keyword_metier.append(mot_cle_job)

    #S'il n'y a plus de page à scraper, on stoppe le code
    try: #block à exécuter si on a accès à la page suivante
        driver.find_element_by_link_text('Suivant »').click()
        time.sleep(seconds)
    except: #block à exécuter si on se trouve sur la dernière page des résultats
        # ---------------------------- DEBUT CONVERSION EN CSV --------------------------------
        # création de tuples depuis les listes temporaires
        mon_tuples = list(
            zip(id, meta1, meta2, titre_poste, entreprise, localisation, contrat, salaire, date_publication,
                keyword_metier, keyword_localisation, description))

        # conversion dataframe en csv
        df = pd.DataFrame(mon_tuples,
                          columns=['id', 'meta1', 'meta2', 'titre_poste', 'entreprise', 'localisation', 'contrat',
                                   'salaire', 'date_publication', 'keyword_metier', 'keyword_localisation',
                                   'description'])  # conversion tuples en dataframe
        df.to_csv('./indeed_scrap_'+ datetime.datetime.now().strftime("%y-%m-%d-%H-%M") +'.csv')#on inclut le datetime dans le nom du fichier pour éviter qu'un autre scraping l'écrase
        # ---------------------------- FIN CONVERSION EN CSV --------------------------------

        is_last_page = True #on se trouve sur la dernière page des résultats, on stoppe le scraping
        driver.quit()
        sys.exit("Scraping effectué avec succès!")
        return

    return

# Lancement de la recherche de job
recherche_xpath('//*[@id="text-input-what"]', 2, mot_cle_job, 0.5)  # Entrée de la localisation souhaitée
recherche_xpath('//*[@id="text-input-where"]', 3, mot_cle_localisation, 0.5)  # Entrée de la localisation souhaitée + lancement de la recherche
time.sleep(seconds)

if(sort_by_date):
    driver.get(driver.current_url + "&radius=100&limit=100&sort=date")  # augmentation du nombre de résultats par page + rayon de 100km + sort by date
else:
    driver.get(driver.current_url + "&radius=100&limit=100")  # augmentation du nombre de résultats par page + rayon de 100km

# Tant qu on ne se trouve pas sur la dernière page, on laisse le programme scraper les données
while is_last_page == False:
    scrap_current_page()