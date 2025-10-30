import streamlit as st
import os
import pandas as pd
import base64,random
import time,datetime
#libraries to parse the resume pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io,random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
import subprocess, sys
import spacy
import re
import docx2txt
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
# Force PAFY to use internal backend to avoid youtube-dl dependency errors
os.environ.setdefault('PAFY_BACKEND', 'internal')
try:
    import pafy  # for uploading youtube videos
except Exception:
    pafy = None
import plotly.express as px #to create visualisations at the admin session
import nltk
nltk.download('stopwords')



def ensure_spacy_model():
    try:
        # Works if model already installed
        spacy.load('en_core_web_sm')
        return True
    except OSError:
        pass
    # Try importing package directly
    try:
        import en_core_web_sm  # noqa: F401
        spacy.load('en_core_web_sm')
        return True
    except Exception:
        pass
    # Last resort: install the spaCy v2 model compatible with pyresparser
    try:
        model_url = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz'
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', model_url])
        spacy.load('en_core_web_sm')
        return True
    except Exception as e:
        return False

def _fallback_extract_text(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            return pdf_reader(file_path)
        elif file_path.lower().endswith('.docx'):
            return docx2txt.process(file_path) or ""
        else:
            return ""
    except Exception:
        return ""

def _fallback_parse_resume(file_path):
    text = _fallback_extract_text(file_path)
    # Basic patterns
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s\-()]{7,}\d)", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = ""
    if lines:
        first = lines[0]
        if 2 <= len(first.split()) <= 5:
            name = first
    # Keywords to approximate skills
    ds_keyword = ['tensorflow','keras','pytorch','machine learning','ml','deep learning','nlp','llm','flask','fastapi','streamlit','scikit-learn','sklearn','pandas','numpy','scipy','matplotlib','seaborn','xgboost','lightgbm','catboost','opencv','airflow','spark','hadoop','sql','postgres','mysql','snowflake','bigquery']
    web_keyword = ['react','reactjs','nextjs','vue','angular','django','flask','fastapi','node','node js','express','php','laravel','magento','wordpress','javascript','typescript','tailwind','bootstrap','graphql','rest','sdk','html','css']
    android_keyword = ['android','android development','kotlin','java','flutter','dart','xml','jetpack','compose','kivy']
    ios_keyword = ['ios','ios development','swift','swiftui','objective-c','cocoa','cocoa touch','xcode','uikit','core data']
    uiux_keyword = ['ux','ui','adobe xd','figma','zeplin','balsamiq','prototyping','wireframes','storyframes','adobe photoshop','photoshop','illustrator','after effects','premier pro','indesign','user research','user experience','journey map','wireframe','persona','usability']
    vocab = set([*ds_keyword, *web_keyword, *android_keyword, *ios_keyword, *uiux_keyword])
    text_lower = ' ' + re.sub(r'[^a-z0-9+#./ ]+', ' ', text.lower()) + ' '
    skills = []
    for kw in sorted(vocab, key=len, reverse=True):
        pattern = r'(?<![a-z0-9+#./])' + re.escape(kw) + r'(?![a-z0-9+#./])'
        if re.search(pattern, text_lower):
            skills.append(kw)
    skills = sorted(set(skills))
    # Page estimation
    pages = 1
    if file_path.lower().endswith('.pdf'):
        try:
            with open(file_path, 'rb') as fh:
                pages = sum(1 for _ in PDFPage.get_pages(fh, caching=True, check_extractable=True))
        except Exception:
            pages = 1
    data = {
        'name': name or "",
        'email': email_match.group(0) if email_match else "",
        'mobile_number': phone_match.group(0) if phone_match else "",
        'skills': skills,
        'no_of_pages': pages
    }
    return data, text

def safe_parse_resume(file_path):
    # Try pyresparser only if spaCy model is available; otherwise fallback
    try:
        if ensure_spacy_model():
            data = ResumeParser(file_path).get_extracted_data()
            text = _fallback_extract_text(file_path)
            return data or {}, text
    except Exception:
        pass
    return _fallback_parse_resume(file_path)

def fetch_yt_video(link):
    try:
        if pafy is None:
            return "Resume Writing Tips Video"
        video = pafy.new(link)
        return video.title
    except Exception:
        return "Resume Writing Tips Video"

def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

@st.cache_data(show_spinner=False)
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course





#CONNECT TO DATABASE

def get_connection():
    try:
        host = os.getenv('DB_HOST', 'localhost')
        user = os.getenv('DB_USER', 'root')
        password = os.getenv('DB_PASS', 'Manozya@12345')
        db_name = os.getenv('DB_NAME', 'cv')
        return pymysql.connect(host=host, user=user, password=password, db=db_name)
    except Exception:
        st.warning("Database is not reachable. Analytics and saving to DB will be disabled.")
        return None

def insert_data(name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses):
    conn = get_connection()
    if not conn:
        return
    try:
        DB_table_name = 'user_data'
        insert_sql = "insert into " + DB_table_name + """
        values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        rec_values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses)
        with conn.cursor() as cursor:
            cursor.execute(insert_sql, rec_values)
        conn.commit()
    finally:
        conn.close()

st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='./Logo/logo2.png',
)
def run():
    # Theme adjustments and contrast fixes
    st.markdown(
        """
        <style>
        /* Improve contrast in dark mode and unify text color */
        .stMarkdown, .stText, .stHeader, .stSubheader, .stSuccess, .stWarning, .stInfo, div, p, span {
            color: inherit !important;
        }
        /* Streamlit tags input contrast */
        .stTags .tag { background-color: #375a7f22 !important; color: inherit !important; }
        /* Progress bar color */
        .stProgress > div > div > div > div { background-color: #d73b5c !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    img = Image.open('./Logo/logo2.png')
    # img = img.resize((250,250))
    st.image(img)
    st.title("AI Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)


    # Create the DB and table (if DB is available)
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
                cursor.execute(db_sql)
                DB_table_name = 'user_data'
                table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                                (ID INT NOT NULL AUTO_INCREMENT,
                                 Name varchar(500) NOT NULL,
                                 Email_ID VARCHAR(500) NOT NULL,
                                 resume_score VARCHAR(8) NOT NULL,
                                 Timestamp VARCHAR(50) NOT NULL,
                                 Page_no VARCHAR(5) NOT NULL,
                                 Predicted_Field BLOB NOT NULL,
                                 User_level BLOB NOT NULL,
                                 Actual_skills BLOB NOT NULL,
                                 Recommended_skills BLOB NOT NULL,
                                 Recommended_courses BLOB NOT NULL,
                                 PRIMARY KEY (ID));
                                """
                cursor.execute(table_sql)
            conn.commit()
        finally:
            conn.close()
    else:
        st.info("Running without database connection.")
    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose your Resume", type=["pdf","docx"], accept_multiple_files=False)
        if uploaded is not None:
            # Size guard: 5 MB
            if getattr(uploaded, 'size', 0) and uploaded.size > 5 * 1024 * 1024:
                st.error("File too large. Please upload a file under 5 MB.")
                return
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+uploaded.name
            with open(save_image_path, "wb") as f:
                f.write(uploaded.getbuffer())
            if uploaded.type == 'application/pdf' or uploaded.name.lower().endswith('.pdf'):
                show_pdf(save_image_path)
            # Visible progress for extraction
            with st.status("Extracting resume details...", expanded=True) as status:
                st.write("Reading document...")
                prog = st.progress(10)
                time.sleep(0.2)
                st.write("Detecting entities and skills...")
                prog.progress(45)
                resume_data, resume_text = safe_parse_resume(save_image_path)
                prog.progress(90)
                st.write("Finalizing analysis...")
                time.sleep(0.2)
                prog.progress(100)
                status.update(label="Extraction complete", state="complete")
            if resume_data:
                ## Get the whole resume data

                st.header("**Resume Analysis**")
                # Optional JD gap analysis input
                with st.expander("Optional: Compare against a Job Description"):
                    jd_text = st.text_area("Paste Job Description text", height=150)
                    jd_file = st.file_uploader("...or upload JD as .txt/.pdf/.docx", type=["txt","pdf","docx"], accept_multiple_files=False, key="jd")
                    if jd_file is not None and not jd_text:
                        tmp_jd = './Uploaded_Resumes/_jd_'+jd_file.name
                        with open(tmp_jd, 'wb') as jf:
                            jf.write(jd_file.getbuffer())
                        if jd_file.name.lower().endswith('.pdf'):
                            jd_text = pdf_reader(tmp_jd)
                        elif jd_file.name.lower().endswith('.docx'):
                            jd_text = docx2txt.process(tmp_jd) or ""
                        else:
                            jd_text = jd_file.getvalue().decode(errors='ignore')

                st.success("Hello "+ (resume_data['name'] or ''))
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: '+ (resume_data['name'] or ''))
                    st.text('Email: ' + (resume_data['email'] or ''))
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >=3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)

                # st.subheader("**Skills Recommendation💡**")
                ## Skill shows
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',
                    value=resume_data['skills'],key = '1  ')

                ##  keywords
                ds_keyword = ['tensorflow','keras','pytorch','machine learning','ml','deep learning','nlp','llm','flask','fastapi','streamlit','scikit-learn','sklearn','pandas','numpy','scipy','matplotlib','seaborn','xgboost','lightgbm','catboost','opencv','airflow','spark','hadoop','sql','postgres','mysql','powerbi','tableau']
                web_keyword = ['react','react js','reactjs','nextjs','vue','angular','django','flask','fastapi','node','node js','express','php','laravel','magento','wordpress','javascript','typescript','tailwind','bootstrap','graphql','rest','sdk','html','css','c#','.net']
                android_keyword = ['android','android development','kotlin','java','flutter','dart','xml','jetpack','compose','kivy']
                ios_keyword = ['ios','ios development','swift','swiftui','objective-c','cocoa','cocoa touch','xcode','uikit','core data']
                uiux_keyword = ['ux','adobe xd','figma','zeplin','balsamiq','ui','prototyping','wireframes','storyframes','adobe photoshop','photoshop','editing','adobe illustrator','illustrator','after effects','premier pro','indesign','wireframe','solid','grasp','user research','user experience','journey map','persona','usability']

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                ## Score categories by overlap rather than first match
                resume_skills_lower = {s.lower() for s in (resume_data.get('skills') or [])}
                cat_to_keywords = {
                    'Data Science': set(ds_keyword),
                    'Web Development': set(web_keyword),
                    'Android Development': set(android_keyword),
                    'IOS Development': set(ios_keyword),
                    'UI-UX Development': set(uiux_keyword),
                }
                cat_scores = {}
                for cat, vocab in cat_to_keywords.items():
                    cat_scores[cat] = len(resume_skills_lower & {v.lower() for v in vocab})
                
                # If JD provided, bias scores toward JD overlap
                jd_bias = {}
                if 'jd_text' in locals() and jd_text:
                    jd_lower_tmp = ' ' + re.sub(r'[^a-z0-9+#./ ]+', ' ', jd_text.lower()) + ' '
                    for cat, vocab in cat_to_keywords.items():
                        overlap = 0
                        for kw in vocab:
                            pattern = r'(?<![a-z0-9+#./])' + re.escape(kw.lower()) + r'(?![a-z0-9+#./])'
                            if re.search(pattern, jd_lower_tmp):
                                overlap += 1
                        jd_bias[cat] = 0.5 * overlap
                        cat_scores[cat] = cat_scores.get(cat, 0) + jd_bias[cat]

                reco_field = max(cat_scores.items(), key=lambda kv: kv[1])[0] if cat_scores else ''

                if reco_field == 'Data Science':
                    st.success("** Our analysis says you are looking for Data Science Jobs.**")
                    recommended_skills = ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining','Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping','ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow','Flask','Streamlit']
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost chances of getting a job</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(ds_course)
                elif reco_field == 'Web Development':
                    st.success("** Our analysis says you are looking for Web Development Jobs **")
                    recommended_skills = ['React','Django','Node JS','React JS','PHP','Laravel','Magento','Wordpress','Javascript','Angular JS','C#','Flask','SDK']
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='3')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost chances of getting a job</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(web_course)
                elif reco_field == 'Android Development':
                    st.success("** Our analysis says you are looking for Android App Development Jobs **")
                    recommended_skills = ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy','GIT','SDK','SQLite']
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='4')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost chances of getting a job</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(android_course)
                elif reco_field == 'IOS Development':
                    st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                    recommended_skills = ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C','SQLite','Plist','StoreKit','UI-Kit','AV Foundation','Auto-Layout']
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='5')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost chances of getting a job</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(ios_course)
                elif reco_field == 'UI-UX Development':
                    st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                    recommended_skills = ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping','Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator','After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='6')
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost chances of getting a job</h4>''', unsafe_allow_html=True)
                    rec_course = course_recommender(uiux_course)

                # Projects recommendation based on domain and skills
                st.subheader("**Project Ideas Based on Your Profile**")
                proj_ideas = []
                if reco_field == 'Data Science':
                    proj_ideas = [
                        ("Customer Churn Prediction", "End-to-end ML pipeline with feature engineering and model monitoring."),
                        ("Sales Forecasting with XGBoost", "Time-series forecasting with cross-validation and SHAP explainability."),
                        ("Resume Parser API", "NER-based skill extraction service with FastAPI and CI tests."),
                        ("Anomaly Detection on Transactions", "Isolation Forest/Autoencoder with drift alerts via Airflow."),
                    ]
                elif reco_field == 'Web Development':
                    proj_ideas = [
                        ("Job Board with Search", "Full-stack app (React + Django/Node) with auth and Elastic search."),
                        ("Personal Analytics Dashboard", "Next.js dashboard pulling GitHub/Google Analytics APIs."),
                        ("E-commerce Cart & Checkout", "Stripe integration, serverless webhooks, order management."),
                    ]
                elif reco_field == 'Android Development':
                    proj_ideas = [
                        ("Habit Tracker", "Jetpack Compose, Room DB, WorkManager notifications."),
                        ("News Reader App", "MVVM, Retrofit, offline cache, pagination."),
                    ]
                elif reco_field == 'IOS Development':
                    proj_ideas = [
                        ("Fitness Planner", "SwiftUI, HealthKit integration, CoreData syncing."),
                        ("Recipe App", "Combine + async networking, share extensions."),
                    ]
                elif reco_field == 'UI-UX Development':
                    proj_ideas = [
                        ("Design System Library", "Figma library with tokens and accessible components."),
                        ("Onboarding Flow Redesign", "Wireframes to hi-fi prototype with usability test results."),
                    ]

                # Add a few skill-tailored ideas
                lower_skills = {s.lower() for s in (resume_data.get('skills') or [])}
                if 'nlp' in lower_skills or 'llm' in lower_skills:
                    proj_ideas.append(("LLM-Powered Q&A Bot", "Retrieval augmented generation using vector DB and evaluation."))
                if 'fastapi' in lower_skills or 'flask' in lower_skills:
                    proj_ideas.append(("Microservice for Feature Store", "Serve features to models with caching and auth."))
                if 'graphql' in lower_skills:
                    proj_ideas.append(("GraphQL API for Resume Data", "Schema-first API with caching and resolvers."))

                if proj_ideas:
                    for idx, (title, desc) in enumerate(proj_ideas[:6], start=1):
                        st.markdown(f"{idx}. **{title}** — {desc}")

                
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                if 'Objective' in resume_text:
                    resume_score = resume_score+20
                    st.markdown('''<h5 style='text-align: left;'>[+] Awesome! You have added Objective</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left;'>[-] Please add your career objective, it will give your career intention to the Recruiters.</h4>''',unsafe_allow_html=True)

                if 'Declaration'  in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left;'>[+] Awesome! You have added Declaration</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left;'>[-] Please add Declaration. It will give assurance that everything written on your resume is true and acknowledged by you.</h4>''',unsafe_allow_html=True)

                if ('Hobbies' in resume_text) or ('Interests' in resume_text):
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left;'>[-] Please add Hobbies. It will show your personality to recruiters and indicate role fit.</h4>''',unsafe_allow_html=True)

                if 'Achievements' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left;'>[-] Please add Projects. It will show relevant hands-on work.</h4>''',unsafe_allow_html=True)

                st.subheader("**Resume Score📝**")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")

                insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                              str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                              str(recommended_skills), str(rec_course))

                # JD Gap Analysis (if JD present)
                if 'jd_text' in locals() and jd_text:
                    st.subheader("Job Description Match Analysis")
                    jd_lower = ' ' + re.sub(r'[^a-z0-9+#./ ]+', ' ', jd_text.lower()) + ' '
                    extracted = {s.lower() for s in (resume_data.get('skills') or [])}
                    match_vocab = list(set(ds_keyword + web_keyword + android_keyword + ios_keyword + uiux_keyword))
                    jd_skills = []
                    for kw in sorted(match_vocab, key=len, reverse=True):
                        pattern = r'(?<![a-z0-9+#./])' + re.escape(kw) + r'(?![a-z0-9+#./])'
                        if re.search(pattern, jd_lower):
                            jd_skills.append(kw)
                    jd_skills = sorted(set(jd_skills))
                    matched = sorted({k for k in jd_skills if k in extracted})
                    missing = sorted([k for k in jd_skills if k not in extracted])
                    total = len(jd_skills)
                    cov = int(round((len(matched) / total) * 100)) if total else 0

                    # Summary KPIs
                    st.markdown(f"**JD skills identified:** {total}  |  **Matched:** {len(matched)}  |  **Missing:** {len(missing)}  |  **Coverage:** {cov}%")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Matched Skills**")
                        if matched:
                            st.markdown("\n".join([f"- {m}" for m in matched]))
                        else:
                            st.markdown("- —")
                    with c2:
                        st.markdown("**Missing Skills**")
                        if missing:
                            st.markdown("\n".join([f"- {m}" for m in missing]))
                        else:
                            st.markdown("- —")

                    # Suggestions
                    if missing:
                        st.info("Consider addressing missing skills in your resume where relevant. Prioritize the top items that appear frequently in the JD.")

                    # Smart JD -> Resume Tailoring
                    st.subheader("Smart JD Tailoring")
                    # Try to infer a role from JD (first line or presence of keywords)
                    jd_lines = [ln.strip() for ln in (jd_text or '').splitlines() if ln.strip()]
                    inferred_role = jd_lines[0][:80] if jd_lines else reco_field
                    # Choose up to 5 priority skills (missing first, then matched)
                    priority = (missing[:5] if missing else []) + [m for m in matched if m not in missing][:5]
                    st.markdown("**Tailored Summary (suggestion):**")
                    summary_skills = ", ".join(priority[:4]) if priority else ", ".join(matched[:4])
                    st.write(f"Experienced candidate targeting {inferred_role} with strengths in {summary_skills or 'core role skills'}, delivering measurable impact through data-driven solutions and cross-functional collaboration.")

                    st.markdown("**Tailored Bullets (suggestions):**")
                    bullet_templates = []
                    for sk in priority[:5]:
                        bullet_templates.append(f"Implemented {sk}-driven solution achieving X% improvement in key metric; leveraged best practices and tooling to ensure reliability and scalability.")
                    if not bullet_templates:
                        bullet_templates = [
                            "Delivered end-to-end outcomes aligned with JD requirements; quantified impact with clear metrics and ownership.",
                            "Collaborated across teams to meet acceptance criteria; automated testing/monitoring to ensure quality."
                        ]
                    for b in bullet_templates:
                        st.markdown(f"- {b}")

                    st.caption("Tip: Replace placeholders like X% with real metrics and reference specific projects above.")

                # ATS Optimization Score & Breakdown
                st.subheader("ATS Optimization Score")
                # Components
                rt_lower = (resume_text or '').lower()
                sections = {
                    'summary/objective': any(k in rt_lower for k in ['objective','summary','profile']),
                    'experience': 'experience' in rt_lower or 'work history' in rt_lower,
                    'education': 'education' in rt_lower,
                    'skills': 'skill' in rt_lower,
                    'projects': 'project' in rt_lower,
                    'certifications': 'certification' in rt_lower or 'certificate' in rt_lower,
                }
                section_score = (sum(1 for v in sections.values() if v) / len(sections)) * 30.0
                # JD coverage weight (if JD present)
                jd_cov = cov if ('jd_text' in locals() and jd_text) else 0
                jd_score = min(40.0, jd_cov * 0.4)  # up to 40 points
                # Contact info
                contact_score = 10.0 if (resume_data.get('email') or resume_data.get('mobile_number')) else 0.0
                # Length
                pages = int(resume_data.get('no_of_pages') or 1)
                length_score = 10.0 if 1 <= pages <= 3 else 5.0 if pages == 4 else 0.0
                # File format
                fmt_ok = 1 if (save_image_path.lower().endswith('.pdf') or save_image_path.lower().endswith('.docx')) else 0
                format_score = 10.0 if fmt_ok else 0.0
                ats_total = int(round(section_score + jd_score + contact_score + length_score + format_score))
                st.progress(min(ats_total, 100))
                st.write(f"Overall ATS score: {ats_total}/100")
                with st.expander("View ATS breakdown"):
                    st.markdown(f"- Sections present: {int(section_score)}/30 — " + ", ".join([k for k,v in sections.items() if v]) or '—')
                    st.markdown(f"- JD coverage: {int(jd_score)}/40")
                    st.markdown(f"- Contact info: {int(contact_score)}/10")
                    st.markdown(f"- Length: {int(length_score)}/10 (pages: {pages})")
                    st.markdown(f"- File format: {int(format_score)}/10")

                # Downloadable JSON report
                report = {
                    'name': resume_data.get('name'),
                    'email': resume_data.get('email'),
                    'mobile_number': resume_data.get('mobile_number'),
                    'pages': resume_data.get('no_of_pages'),
                    'field': reco_field,
                    'level': cand_level,
                    'skills': resume_data.get('skills'),
                    'recommended_skills': recommended_skills,
                    'recommended_courses': rec_course,
                    'resume_score': score,
                    'ats_score': ats_total,
                    'timestamp': timestamp,
                }
                import json
                st.download_button(
                    label="Download Analysis (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"resume_analysis_{int(time.time())}.json",
                    mime="application/json",
                )


                ## Resume writing video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                try:
                    resume_vid = random.choice(resume_videos)
                    res_vid_title = fetch_yt_video(resume_vid)
                    st.subheader("✅ **"+res_vid_title+"**")
                    st.video(resume_vid)
                except Exception as e:
                    st.info("📹 Video temporarily unavailable. Please check back later.")



                ## Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                try:
                    interview_vid = random.choice(interview_videos)
                    int_vid_title = fetch_yt_video(interview_vid)
                    st.subheader("✅ **" + int_vid_title + "**")
                    st.video(interview_vid)
                except Exception as e:
                    st.info("📹 Video temporarily unavailable. Please check back later.")

                # Data is saved only if DB connection is available inside insert_data
            else:
                st.error('Something went wrong..')
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            admin_user = os.getenv('ADMIN_USER', 'admin')
            admin_pass = os.getenv('ADMIN_PASS', 'pass')
            if ad_user == admin_user and ad_password == admin_pass:
                st.success("Welcome Admin !")
                
                try:
                    conn = get_connection()
                    if not conn:
                        st.error("Database not available.")
                        return
                    # Display Data
                    with conn.cursor() as cursor:
                        cursor.execute('''SELECT*FROM user_data''')
                        data = cursor.fetchall()
                    
                    if data:
                        st.header("**User's Data**")
                        df = pd.DataFrame(data, columns=['ID', 'Name', 'Email_ID', 'resume_score', 'Timestamp', 'Page_no',
                                                         'Predicted_Field', 'User_level', 'Actual_skills', 'Recommended_skills',
                                                         'Recommended_courses'])
                        st.dataframe(df)
                        st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
                        
                        ## Admin Side Data
                        query = 'select * from user_data;'
                        plot_data = pd.read_sql(query, conn)

                        ## Pie chart for predicted field recommendations
                        if not plot_data.empty and 'Predicted_Field' in plot_data.columns:
                            # Convert BLOB data to strings for visualization
                            plot_data['Predicted_Field'] = plot_data['Predicted_Field'].astype(str)
                            labels = plot_data.Predicted_Field.dropna().unique()
                            if len(labels) > 0:
                                values = plot_data.Predicted_Field.value_counts()
                                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                                fig = px.pie(plot_data, values=values, names=labels, title='Predicted Field according to the Skills')
                                st.plotly_chart(fig)
                            else:
                                st.info("No predicted field data available for visualization")

                        ### Pie chart for User's👨‍💻 Experienced Level
                        if not plot_data.empty and 'User_level' in plot_data.columns:
                            # Convert BLOB data to strings for visualization
                            plot_data['User_level'] = plot_data['User_level'].astype(str)
                            labels = plot_data.User_level.dropna().unique()
                            if len(labels) > 0:
                                values = plot_data.User_level.value_counts()
                                st.subheader("**Pie-Chart for User's Experienced Level**")
                                fig = px.pie(plot_data, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                                st.plotly_chart(fig)
                            else:
                                st.info("No user level data available for visualization")
                    else:
                        st.info("No user data found in the database. Upload some resumes to see analytics!")
                        
                except Exception as e:
                    st.error(f"Error accessing database: {str(e)}")
                    st.info("Please ensure MySQL is running and the 'cv' database exists.")
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass


            else:
                st.error("Wrong ID & Password Provided")
run()