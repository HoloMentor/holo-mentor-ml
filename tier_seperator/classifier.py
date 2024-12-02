import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from dotenv import load_dotenv
import psycopg2
from datetime import datetime



def classify(df, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(df[['marks']])
 
    df['tier'] = gmm.predict(df[['marks']])
    cluster_means = df.groupby('tier')['marks'].mean()
    sorted_clusters = cluster_means.sort_values(ascending=False).index
    cluster_mapping = {cluster: i + 1 for i, cluster in enumerate(sorted_clusters)}
    df['tier'] = df['tier'].map(cluster_mapping)
    
    return df

def tier_df(df):
    tier_df = df[["student_class_id", "tier"]]
    return df


def update_institute_class_students(conn, update_data):
    cur = conn.cursor()

    for index, row in update_data.iterrows():
        student_id = int(row['student_class_id']) 
        tier = int(row['tier'])  

        update_query = """
        UPDATE institute_class_students
        SET tier = %s, tier_last_updated_at = %s
        WHERE id = %s
        """
        
        current_time = datetime.now()
        
        cur.execute(update_query, (tier, current_time, student_id))

    conn.commit()
    cur.close()


def insert_and_update_institute_class_tier_students(conn, insert_data, out_of, class_id):
    cur = conn.cursor()

    for index, row in insert_data.iterrows():
        student_id = int(row['student_class_id']) 
        tier = int(row['tier'])
        marks = float(row['marks'])  

        # Fetch class_id and institute_id for the student
        select_query = """
            SELECT class_id, institute_id 
            FROM institute_class_students 
            WHERE id = %s
        """
        cur.execute(select_query, (student_id, class_id))
        result = cur.fetchone()

        if result is not None:
            institute_id = result[1]

            # Fetch the correct student_class_id
            select_student_class_id_query = """
                SELECT id
                FROM institute_class_students
                WHERE id = %s
            """
            cur.execute(select_student_class_id_query, (student_id, class_id))
            result_1 = cur.fetchone()

            if result_1 is not None:
                student_class_id = result_1[0]  # Correct ID from institute_class_students

                # Insert into institute_class_tier_students
                insert_query = """
                    INSERT INTO institute_class_tier_students (student_class_id, class_id, institute_id, tier, marks, marks_out_of, upload_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) 
                """
                cur.execute(insert_query, (student_class_id, class_id, institute_id, tier, marks, out_of, datetime.now()))

                # Update institute_class_students
                update_query = """
                    UPDATE institute_class_students
                    SET tier = %s, tier_last_updated_at = %s
                    WHERE id = %s
                """
                cur.execute(update_query, (tier, datetime.now(), student_class_id))
        
    conn.commit()
    cur.close()


def insert_institute_notification(conn,class_id):

    cur = conn.cursor()

    institute_id_query = """
        SELECT institute_id
        FROM institute_classes
        WHERE id = %s
    """

    cur.execute(institute_id_query, (class_id,))
    result = cur.fetchone()
    institute_id = result[0]

    insert_query = """INSERT INTO institute_class_notifications (institute_id, institute_class_id, title, message, created_at)
                        VALUES (%s, %s, %s, %s, %s)"""

    title = f"Tier Classification Updated"
    message = f"Tier Classification updaetd for Class {class_id} in Institute {institute_id} at {datetime.now()}"

    cur.execute(insert_query, (institute_id, class_id, title, message, datetime.now()))

    conn.commit()
    cur.close()