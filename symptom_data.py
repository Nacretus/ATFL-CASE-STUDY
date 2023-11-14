#from model import prediction
dis_symp = {    
    'Abscess': ['redness', 'testicular pain', 'swelling'],
    'Acquired Capillary Haemangioma of Eyelid': ['raised red blue lesion'],
    'Acquired Immuno Deficiency Syndrome': ['flu like illness'],
    'Acute encephalitis syndrome': ['vomiting', 'fever', 'headache', 'confusion', 'stiff neck'],
    'Adult Inclusion Conjunctivitis': ['scratchiness', 'reddish eye'],
    'Alcohol Abuse and Alcoholism': ['difficulty cutting', 'acquiring drinking alcohol taking lot time', 'usage resulting problem', 'drinking large amount alcohol long period', 'withdrawal occurring stopping'],
    'Alopecia (hair loss)': ['loss hair part head body'],
    'Alzheimer': ['difficulty remembering recent event', 'problem language', 'mood swing', 'disorientation'],
    'Amaurosis Fugax': ['temporary fleeting vision one eye'],
    'Amblyopia': ['decreased vision'],
    'Amoebiasis': ['bloody diarrhea', 'testicular pain'],
    'Anaemia': ['muscle weakness', 'feeling tired', 'feeling like passing', 'shortness breath'],
    'Aniseikonia': ['object different size eye'],
    'Anisometropia': ['one eye myopia eye hyperopia'],
    'Antepartum hemorrhage (Bleeding in late pregnancy)': ['increased breath rate', 'loss lot blood childbirth', 'increased heart rate', 'feeling faint upon standing'],
    'Anthrax': ['vomiting', 'fever', 'nausea', 'chest pain', 'diarrhea', 'shortness breath', 'testicular pain', 'abscess', 'small blister surrounding swelling'],
    'Anxiety': ['fast heart rate', 'worrying', 'shakiness'],
    'Appendicitis': ['vomiting', 'decreased appetite', 'right lower abdominal pain'],
    'Arthritis': ['redness', 'decreased range motion', 'swelling', 'stiffness', 'joint bone pain'],
    'Asbestos-related diseases': ['barky cough', 'wheezing', 'shortness breath', 'chest pain'],
    'Aseptic meningitis': ['fever', 'neck stiffness', 'headache'],
    'Asthma': ['shortness breath', 'recurring episode wheezing', 'chest tightness', 'coughing'],
    'Astigmatism': ['headache', 'distorted blurred vision distance', 'eyestrain'],
    'Atrophy': ['progressive muscle weakness'],
    'Autism': ['trouble social interaction', 'impaired communication', 'restricted interest', 'repetitive behavior'],
    'Bad Breath (Halitosis)': ['unpleasant smell present breath'],
    "Bell's Palsy": ['change taste', 'inability move facial muscle one side', 'pain around ear'],
    'Beriberi': ['shortness breath', 'fast heart rate', 'wet', 'leg swelling'],
    'Black Death': ['fever', 'muscle weakness', 'headache'],
    'Bleeding Gums': ['bad breath', 'red', 'swollen', 'painful', 'loose teeth', 'bleeding gum'],
    'Blindness': ['decreased ability see'],
    'Botulism': ['muscle weakness', 'trouble seeing', 'trouble speaking', 'feeling tired'],
    'Brain Tumour': ['vomiting', 'vary depending part brain involved', 'mental change', 'headache', 'problem vision', 'seizure'],
    'Breast Cancer / Carcinoma': ['change breast shape', 'dimpling skin', 'fluid nipple', 'lump breast', 'newly inverted nipple', 'red scaly patch skin breast'],
    'Bronchitis': ['chest discomfort', 'wheezing', 'shortness breath', 'coughing mucus'],
    'Brucellosis': ['coughing'],
    'Bubonic plague': ['vomiting', 'fever', 'headache', 'swollen lymph node'],
    'Bunion': ['red', 'painful joint base big toe', 'prominent'],
    'Burns': ['red without blister'],
    'Calculi': ['vomiting', 'blood urine', 'severe pain lower back abdomen', 'nausea'],
    'Campylobacter infection': ['nausea', 'testicular pain'],
    'Cancer': ['abnormal bleeding', 'change bowel movement', 'lump breast', 'prolonged cough', 'unexplained weight loss'],
    'Candidiasis': ['white patch vaginal discharge', 'itchy'],
    'Carbon monoxide poisoning': ['vomiting', 'headache', 'muscle weakness', 'confusion', 'dizziness', 'chest pain'],
    'Carpal Tunnel Syndrome': ['half ring finger', 'tingling thumb', 'testicular pain', 'numbness', 'weak grip', 'middle finger', 'index'],
    'Cavities': ['difficulty eating', 'tooth loss', 'testicular pain'],
    'Celiacs disease': ['diarrhoea', 'dermatitis herpetiformis', 'abdominal distention', 'unintended weight loss', 'malabsorption', 'constipation', 'none non specific'],
    'Cerebral palsy': ['stiff muscle', 'poor coordination', 'tremor', 'weak muscle'],
    'Chagas disease': ['fever', 'headache', 'large lymph node'],
    'Chalazion': ['non painful cyst middle eyelid', 'red'],
    'Chickenpox': ['fever', 'loss appetite', 'tiredness', 'headache', 'small', 'itchy blister'],
    'Chikungunya Fever': ['fever', 'joint bone pain'],
    'Childhood Exotropia': ['nonaligned eye'],
    'Chlamydia': ['discharge penis', 'burning urination', 'vaginal discharge'],
    'Cholera': ['vomiting', 'muscle cramp', 'large amount watery diarrhea'],
    'Chorea': ['mental ability', 'coordination', 'jerky body movement', 'problem mood'],
    'Chronic fatigue syndrome': ['long term fatigue', 'others'],
    'Chronic obstructive pulmonary disease (COPD)': ['shortness breath', 'cough sputum production'],
    'Cleft Lip and Cleft Palate': ['opening upper lip may extend nose palate'],
    'Colitis': ['fever', 'diarrhea mixed blood', 'anemia', 'unintended weight loss', 'testicular pain'],
    'Colorectal Cancer': ['unintended weight loss', 'blood stool', 'change bowel movement', 'feeling tired time'],
    'Common cold': ['barky cough', 'fever', 'runny nose', 'sore throat'],
    'Condyloma': ['skin lesion generally pink color project outward'],
    'Congenital anomalies (birth defects)': ['intellectual disability', 'developmental disability', 'physical disability'],
    'Congestive heart disease': ['feeling tired', 'shortness breath', 'leg swelling'],
    'Corneal Abrasion': ['light sensitivity', 'eye pain'],
    'Coronary Heart Disease': ['chest pain', 'shortness breath'],
    'Coronavirus disease 2019 (COVID-19)': ['fever', 'barky cough', 'shortness breath', 'sometimes symptom', 'loss smell', 'fatigue'],
    'Cough': ['runny nose', 'fever', 'barky cough'],
    'Crimean Congo haemorrhagic fever (CCHF)': ['vomiting', 'fever', 'headache', 'diarrhea', 'testicular pain', 'bleeding skin'],
    'Dehydration': ['nausea', 'headache', 'dizziness', 'profuse sweating', 'fatigue'],
    'Dementia': ['problem language', 'emotional problem', 'decreased motivation', 'decreased ability think remember'],
    'Dengue': ['fever', 'muscle joint pain', 'headache', 'maculopapular rash'],
    'Diabetes Mellitus': ['frequent urination', 'increased hunger', 'increased thirst'],
    'Diabetic Retinopathy': ['vision loss', 'blurry vision', 'may symptom', 'blindness'],
    'Diarrhea': ['loose frequent bowel movement', 'dehydration'],
    'Diphtheria': ['barky cough', 'fever', 'sore throat'],
    "Down's Syndrome": ['characteristic facial feature', 'mild moderate intellectual disability', 'delayed physical growth'],
    'Dracunculiasis (guinea-worm disease)': ['painful blister lower leg'],
    'Dysentery': ['fever', 'bloody diarrhea', 'testicular pain'],
    'Ear infection': ['fever', 'ear pain', 'hearing loss'],
    'Early pregnancy loss': ['vaginal bleeding without pain'],
    'Ebola': ['diarrhoea', 'fever', 'headache', 'vaginal bleeding', 'muscular pain', 'sore throat'],
    'Eclampsia': ['high blood pressure', 'seizure'],
    'Ectopic pregnancy': ['testicular pain', 'vaginal bleeding'],
    'Eczema': ['maculopapular rash', 'itchiness', 'red skin'],
    'Endometriosis': ['infertility', 'testicular pain'],
    'Epilepsy': ['period vigorous shaking', 'nearly undetectable spell'],
    'Fibroids': ['painful heavy period'],
    'Fibromyalgia': ['sleep problem', 'feeling tired', 'widespread pain'],
    'Food Poisoning': ['vomiting', 'fever', 'diarrhea', 'abdominal cramp'],
    'Frost Bite': ['pale color', 'feeling cold', 'clumsy', 'numbness'],
    'GERD': ['breathing problem', 'bad breath', 'taste acid', 'heartburn', 'chest pain'],
    'Gaming disorder': ['social withdrawal', 'depression', 'playing video game extremely long period time'],
    'Gangrene': ['skin breakdown', 'testicular pain', 'numbness', 'coolness', 'change skin color red black'],
    'Gastroenteritis': ['vomiting', 'fever', 'diarrhea', 'testicular pain'],
    'Genital herpes': ['flu like symptom', 'small blister break open form painful ulcer'],
    'Glaucoma': ['nausea', 'eye pain', 'mid dilated pupil', 'vision loss', 'redness eye'],
    'Goitre': ['sleeping problem', 'irritability', 'muscle weakness', 'unintended weight loss', 'enlarged thyroid', 'poor tolerance heat', 'fast heartbeat'],
    'Gonorrhea': ['discharge penis', 'burning urination', 'testicular pain', 'vaginal discharge'],
    'Guillain-Barré syndrome': ['muscle weakness beginning foot hand'],
    'Haemophilia': ['easy prolonged bleeding'],
    'Hand, Foot and Mouth Disease': ['fever', 'flat discolored spot bump may blister'],
    'Heat-Related Illnesses and Heat waves': ['nausea', 'red', 'headache', 'confusion', 'dizziness', 'high body temperature', 'dry damp skin'],
    'Hepatitis': ['yellowish skin', 'testicular pain', 'poor appetite'],
    'Hepatitis A': ['vomiting', 'nausea', 'fever', 'dark urine', 'diarrhea', 'testicular pain', 'jaundice'],
    'Hepatitis B': ['yellowish skin', 'testicular pain', 'dark urine', 'tiredness'],
    'Hepatitis C': ['typically none'],
    'Hepatitis D': ['feeling tired', 'nausea vomiting'],
    'Hepatitis E': ['nausea', 'jaundice'],
    'Herpes Simplex': ['fever', 'small blister break open form painful ulcer', 'swollen lymph node'],
    'High risk pregnancy': ['frequent urination', 'missed period', 'nausea vomiting', 'tender breast', 'increased hunger'],
    'Human papillomavirus': ['wart'],
    'Hypermetropia': ['close object appear blurry'],
    'Hyperthyroidism': ['enlargement thyroid', 'sleeping problem', 'irritability', 'muscle weakness', 'diarrhea', 'unintended weight loss', 'heat intolerance', 'fast heartbeat'],
    'Hypothyroid': ['weight gain', 'constipation', 'feeling tired', 'poor ability tolerate cold', 'depression'],
    'Hypotonia': ['muscle weakness'],
    'Impetigo': ['yellowish skin crust', 'painful'],
    'Inflammatory Bowel Disease': ['fever', 'unintended weight loss', 'testicular pain', 'diarrhea may bloody'],
    'Influenza': ['fever', 'muscle joint pain', 'headache', 'coughing', 'feeling tired', 'sore throat', 'runny nose'],
    'Insomnia': ['irritability', 'trouble sleeping', 'low energy', 'daytime sleepiness', 'depressed mood'],
    'Interstitial cystitis': ['chronic pain bladder', 'needing urinate often', 'feeling need urinate right away', 'pain sex'],
    'Iritis': ['blurred vision', 'headache', 'red eye', 'photophobia', 'burning redness eye'],
    'Iron Deficiency Anemia': ['pallor', 'muscle weakness', 'confusion', 'shortness breath', 'feeling tired'],
    'Irritable bowel syndrome': ['diarrhea', 'testicular pain', 'constipation'],
    'Japanese Encephalitis': ['vomiting', 'fever', 'headache', 'confusion', 'seizure'],
    'Jaundice': ['yellowish coloration skin white eye', 'itchiness'],
    'Kala-azar/ Leishmaniasis': ['non itchy skin ulcer', 'fever', 'enlarged thyroid', 'low red blood cell'],
    "Kaposi’s Sarcoma": ['purple colored skin lesion'],
    'Keratoconjunctivitis Sicca (Dry eye syndrome)': ['redness', 'dry eye', 'blurred vision', 'vaginal discharge', 'irritation'],
    'Keratoconus': ['nearsightedness', 'blurry vision', 'light sensitivity'],
    'Kuru': ['gradual loss coordination', 'body tremor', 'random outburst laughter'],
    'Laryngitis': ['hoarse voice', 'fever', 'testicular pain'],
    'Lead poisoning': ['intellectual disability', 'tingling hand foot', 'irritability', 'inability child', 'headache', 'testicular pain', 'constipation', 'memory problem'],
    'Legionellosis': ['fever', 'barky cough', 'headache', 'testicular pain', 'shortness breath'],
    'Leprosy': ['decreased ability feel pain'],
    'Leptospirosis': ['headache', 'testicular pain', 'fever'],
    'Leukemia': ['fever', 'increased risk infection', 'vaginal bleeding', 'feeling tired', 'bruising'],
    'Lice': ['itching result trouble sleeping'],
    'Lung cancer': ['coughing including coughing blood', 'unintended weight loss', 'shortness breath', 'chest pain'],
    'Lupus erythematosus': ['fever', 'painful swollen joint', 'swollen lymph node', 'feeling tired', 'mouth ulcer', 'chest pain', 'hair loss', 'red rash'],
    'Lyme disease': ['expanding area redness site tick bite', 'tiredness', 'headache', 'fever'],
    'Lymphoma': ['fever', 'unintended weight loss', 'sweat', 'feeling tired', 'enlarged lymph node neck', 'itching'],
    'Mad cow disease': ['trouble walking', 'unintended weight loss', 'abnormal behavior', 'unable move'],
    'Malaria': ['vomiting', 'fever', 'headache'],
    'Marburg fever': ['fever', 'muscle weakness', 'myalgia'],
    'Mastitis': ['localized breast pain redness', 'fever'],
    'Measles': ['fever', 'barky cough', 'maculopapular rash', 'inflamed eye', 'runny nose'],
    'Melanoma': ['change color', 'skin breakdown', 'itchiness', 'irregular edge', 'mole increasing size'],
    'Middle East respiratory syndrome coronavirus (MERS‐CoV)': ['barky cough', 'fever', 'shortness breath'],
    'Migraine': ['nausea', 'headache', 'sensitivity smell', 'sensitivity sound', 'light sensitivity'],
    'Mononucleosis': ['enlarged lymph node neck', 'fever', 'tiredness', 'sore throat'],
    'Mouth Breathing': ['bad breath', 'hoarse voice', 'fatigue', 'dry mouth', 'sore throat', 'stuffy itchy nose'],
    'Multiple myeloma': ['anemia', 'frequent infection', 'bone pain', 'vaginal bleeding'],
    'Multiple sclerosis': ['trouble coordination', 'muscle weakness', 'trouble sensation', 'blindness one eye', 'double vision'],
    'Mumps': ['fever', 'feeling generally unwell', 'headache', 'painful swelling parotid gland', 'testicular pain'],
    'Muscular dystrophy': ['trouble walking', 'increasing weakening', 'breakdown skeletal muscle'],
    'Myasthenia gravis': ['varying degree muscle weakness', 'trouble talking', 'trouble walking', 'double vision', 'drooping eyelid'],
    'Myelitis': ['weakness limb'],
    'Myocardial Infarction (Heart Attack)': ['nausea', 'stomach pain', 'cold sweat', 'shortness breath', 'neck', 'feeling tired', 'arm', 'jaw', 'back', 'chest pain', 'feeling faint upon standing'],
    'Myopia': ['distant object appear blurry', 'headache', 'eye strain', 'close object appear blurry'],
    'Narcolepsy': ['sudden loss muscle strength', 'involuntary sleep episode', 'hallucination', 'excessive daytime sleepiness'],
    'Nasal Polyps': ['trouble breathing nose', 'loss smell', 'decreased taste', 'runny nose', 'post nasal drip'],
    'Nausea and Vomiting of Pregnancy and  Hyperemesis gravidarum': ['nausea', 'vomiting', 'weight loss', 'dehydration occur'],
    'Necrotizing Fasciitis': ['severe pain', 'fever', 'purple colored skin affected area'],
    'Neonatal Respiratory Disease Syndrome(NRDS)': ['rapid breathing', 'shortness breath', 'bluish skin coloration'],
    'Neoplasm': ['lump breast'],
    'Neuralgia': ['shock like pain one side face last second minute', 'sudden', 'episode severe'],
    'Nipah virus infection': ['barky cough', 'fever', 'confusion', 'headache'],
    'Obesity': ['increased fat'],
    'Obsessive Compulsive Disorder': ['feel need check thing repeatedly', 'certain thought repeatedly', 'perform certain routine repeatedly'],
    'Oral Cancer': ['persistent rough white red patch mouth lasting longer week', 'difficulty swallowing', 'testicular pain', 'lump bump neck', 'ulceration', 'loose teeth'],
    'Orbital Dermoid': ['painless lump', 'minimal'],
    'Osteoarthritis': ['decreased range motion', 'stiffness', 'joint swelling', 'joint bone pain'],
    'Osteomyelitis': ['pain specific bone', 'fever', 'muscle weakness', 'overlying redness'],
    'Osteoporosis': ['increased risk broken bone'],
    'Paratyphoid fever': ['fever', 'testicular pain', 'headache', 'maculopapular rash'],
    "Parkinson's Disease": ['shaking', 'difficulty walking', 'rigidity', 'slowness movement'],
    'Pelvic inflammatory disease': ['irregular menstruation', 'fever', 'lower abdominal pain', 'pain sex', 'vaginal discharge', 'burning urination'],
    'Perennial Allergic Conjunctivitis': ['watery eye', 'red', 'itchy', 'sneezing', 'swelling around eye', 'stuffy itchy nose', 'itchy ear'],
    'Pericarditis': ['sharp chest pain', 'better sitting worse lying', 'fever'],
    'Peritonitis': ['severe pain', 'fever', 'swelling abdomen'],
    'Pinguecula': ['pinkish', 'triangular tissue growth cornea'],
    'Pneumonia': ['barky cough', 'rapid breathing', 'fever', 'difficulty breathing'],
    'Poliomyelitis': ['muscle weakness resulting inability move'],
    'Polycystic ovary syndrome (PCOS)': ['acne', 'velvety skin', 'patch thick', 'testicular pain', 'difficulty getting pregnant', 'irregular menstrual period', 'excess hair', 'heavy period', 'darker'],
    'Porphyria': ['vomiting', 'fever', 'depending subtype abdominal pain', 'confusion', 'constipation', 'blister sunlight', 'chest pain', 'seizure'],
    'Post Menopausal Bleeding': ['irregular menstruation', 'prolonged', 'abnormally frequent', 'excessive amount uterine bleeding'],
    'Post-herpetic neuralgia': ['pain doesnt go shingle', 'burning stabbing pain'],
    'Postpartum depression/ Perinatal depression': ['cry episode', 'irritability', 'low energy', 'change sleeping eating pattern', 'extreme sadness', 'anxiety'],
    'Preeclampsia': ['high blood pressure', 'protein urine'],
    'Premenstrual syndrome': ['acne', 'feeling tired', 'mood change', 'tender breast', 'bloating'],
    'Presbyopia': ['headache', 'hold reading material farther away', 'hard time reading small print', 'eyestrain'],
    'Preterm birth': ['birth baby younger week gestational age'],
    'Progeria': ['hair loss', 'short height', 'small face', 'growth delay'],
    'Psoriasis': ['scaly patch skin', 'red purple darker skin', 'itchy'],
    'Puerperal sepsis': ['bad smelling vaginal discharge', 'fever', 'lower abdominal pain'],
    'Pulmonary embolism': ['chest pain', 'coughing blood', 'shortness breath'],
    'Ques fever': ['shivering', 'feeling cold'],
    'Quinsy': ['fever', 'change voice', 'testicular pain', 'trouble opening mouth'],
    'Rabies': ['fever', 'fear water', 'trouble sleeping', 'paralysis', 'hallucination', 'confusion', 'excessive salivation', 'coma'],
    "Raynaud's Phenomenon": ['blue', 'red', 'burning', 'affected part turning white'],
    'Repetitive strain injury': ['pulsing pain', 'aching', 'sore wrist', 'tingling', 'extremity weakness'],
    'Rheumatic fever': ['fever', 'erythema marginatum', 'involuntary muscle movement', 'multiple painful joint'],
    'Rheumatism': ['warm', 'painful swollen joint', 'swollen'],
    'Rickets': ['bowed leg', 'trouble sleeping', 'bone pain', 'stunted growth', 'large forehead'],
    'Rift Valley fever': ['fever', 'testicular pain', 'headache'],
    'Rocky Mountain spotted fever': ['fever', 'headache'],
    'Rubella': ['fever', 'maculopapular rash', 'swollen lymph node', 'feeling tired', 'sore throat'],
    'SARS': ['fever', 'dry cough', 'muscle ache difficulty breathing', 'headache'],
    'SIDS': ['death child le one year age'],
    'Sarcoidosis': ['depends organ involved'],
    'Sarcoma': ['swell pain near tumor'],
    'Scabies': ['itchiness', 'pimple like rash'],
    'Scarlet fever': ['fever', 'headache', 'characteristic rash', 'swollen lymph node', 'sore throat'],
    'Schizophrenia': ['hallucination usually hearing voice', 'confused thinking', 'delusion'],
    'Sciatica': ['pain going leg lower back', 'weakness numbness affected leg'],
    'Scrapie': ['delirium', 'insomnia', 'confusion', 'tremor', 'dementia', 'psychosis', 'seizure'],
    'Scrub Typhus': ['fever', 'headache', 'maculopapular rash'],
    'Scurvy': ['change hair', 'muscle weakness', 'feeling tired', 'easy prolonged bleeding', 'gum disease', 'sore arm leg'],
    'Sepsis': ['fever', 'confusion', 'low blood pressure', 'increased breathing rate', 'increased heart rate'],
    'Sexually transmitted infections (STIs)': ['ulcer around genitals', 'testicular pain', 'vaginal discharge'],
    'Shaken Baby Syndrome': ['variable'],
    'Shigellosis': ['fever', 'diarrhea', 'testicular pain'],
    'Shin splints': ['pain along inside edge shinbone'],
    'Shingles': ['painful rash occurring stripe'],
    'Sickle-cell anemia': ['anemia', 'swelling hand foot', 'bacterial infection', 'attack pain', 'stroke'],
    'Smallpox': ['vomiting', 'fever', 'mouth sore', 'fluid filled blister scab'],
    'Stevens-Johnson syndrome': ['fever', 'skin peeling', 'painful skin', 'red eye', 'skin blister'],
    'Stomach ulcers': ['vomiting', 'poor appetite', 'unintended weight loss', 'upper abdominal pain', 'belching'],
    'Strep throat': ['fever', 'large lymph node', 'sore throat'],
    'Stroke': ['problem understanding speaking', 'loss vision one side', 'dizziness', 'inability move feel one side body'],
    'Sub-conjunctival Haemorrhage': ['red spot white eye', 'little pain'],
    'Syphilis': ['non itchy skin ulcer', 'painless', 'firm'],
    'Taeniasis': ['unintended weight loss', 'testicular pain'],
    'Taeniasis/cysticercosis': ['cm lump skin'],
    'Tay-Sachs disease': ['sit', 'decreased ability turn', 'crawl'],
    'Tennis elbow': ['painful tender outer part elbow'],
    'Tetanus': ['fever', 'muscle spasm', 'headache'],
    'Thalassaemia': ['dark urine', 'enlarged spleen', 'yellowish skin', 'pale skin', 'feeling tired'],
    'Tinnitus': ['hearing sound external sound present'],
    'Tonsillitis': ['fever', 'enlargement tonsil', 'trouble swallowing', 'large lymph node around neck', 'sore throat'],
    'Toxic shock syndrome': ['fever', 'low blood pressure', 'skin peeling', 'maculopapular rash'],
    'Trachoma': ['blindness', 'eye pain'],
    'Trichinosis': ['vomiting', 'diarrhea', 'testicular pain'],
    'Trichomoniasis': ['burning urination', 'pain sex', 'itching genital area', 'bad smelling thin vaginal discharge'],
    'Tuberculosis': ['chronic cough', 'cough bloody mucus', 'fever', 'unintended weight loss'],
    'Tularemia': ['non itchy skin ulcer', 'fever', 'large lymph node'],
    'Turners Syndrome': ['short stature', 'swollen hand foot', 'webbed neck'],
    'Urticaria': ['raised', 'itchy bump', 'red'],
    'Varicose Veins': ['fullness', 'pain area'],
    'Vasovagal syncope': ['loss consciousness may sweating', 'decreased ability see', 'ringing ear heartbeat'],
    'Vitamin B12 Deficiency': ['irritability', 'decreased ability think', 'depression', 'change reflex', 'abnormal sensation'],
    'Vitiligo': ['patch white skin'],
    'Warkany syndrome': ['clenched fist overlapping finger', 'small head', 'severe intellectual disability', 'small jaw'],
    'Warts': ['small', 'painless', 'rough skin growth'],
    'Yaws': ['ulcer', 'joint bone pain', 'hard swelling skin'],
    'Yellow Fever': ['fever', 'chill', 'headache', 'testicular pain', 'yellow skin'],
    'Zika virus disease': ['fever', 'red eye', 'headache', 'maculopapular rash', 'joint bone pain'],
    'lactose intolerance': ['gas', 'nausea', 'diarrhea', 'testicular pain', 'bloating'],
    'papilloedema': ['headache', 'problem vision', 'ringing ear heartbeat']
}





all_symptoms = [dis_symp for symptom_list in dis_symp.values() for dis_symp in symptom_list]
unique_symptoms = set(all_symptoms)
sorted_unique_symptoms = sorted(unique_symptoms) #<--- eto yung nag hahawak ng symptoms

#print(sorted_unique_symptoms)
"""all_conditions = list(dis_symp.keys()) # get the keys
sorted_conditions = sorted(all_conditions) # sort keys in alphabetical order

# print each condition with a number before it
for i, condition in enumerate(sorted_conditions, 1):
    print(f"{i}. {condition}")"""

"""def get_prediction_result(input_text):
    # Call the prediction function
    prediction_result = prediction(input_text)

    # Split the result into lines
    lines = prediction_result.split("\n")

    # Initialize an empty string to store the illnesses and symptoms
    illness_symptoms_str = "Top 5 Predicted Conditions:\n"

    # Skip the first line ("Top 5 Predicted Conditions:")
    for line in lines[1:]:
        if line:
            # Extract the illness name from the line
            illness = line.split(". Name of illness: ")[1].split(", Prediction Accuracy:")[0]

            # Look up the symptoms in the dictionary
            symptoms = dis_symp.get(illness, "No symptoms found")

            # Add the illness and symptoms to the string
            illness_symptoms_str += f"{illness}:\nSymptoms: {', '.join(symptoms)}\n"

    return illness_symptoms_str"""



