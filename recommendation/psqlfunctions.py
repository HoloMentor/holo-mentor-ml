import json

def check_custom_quiz_already_exists(class_id, student_id, conn):
    """ Check if a custom quiz already exists for a student in a class this week """
    cur = conn.cursor()

    query = """
                SELECT *
                FROM custom_quiz
                WHERE class_id = %s
                AND user_id = %s
                AND created_at >= NOW() - INTERVAL '7 days';
            """

    cur.execute(query, (class_id, student_id))

    row = cur.fetchone()

    cur.close()

    if row:
        return True
    else:
        return False


def load_wrong_questions(class_id, student_id,conn):
    cur = conn.cursor()

    query = """
                SELECT qa.*
                FROM quiz_answers qa
                JOIN mcq_questions mq ON qa.mcq_question_id = mq.id
                WHERE qa.user_id = %s
                AND mq.class_id = %s
                AND qa.answer != mq.answer
                ORDER BY qa.created_at DESC
                LIMIT 10;
            """


    cur.execute(query, (student_id, class_id))

    rows = cur.fetchall()

    col_names = [desc[0] for desc in cur.description]

    result = [dict(zip(col_names, row)) for row in rows]
    wrong_mcq_question_ids = [q['mcq_question_id'] for q in result]
    cur.close()
    return wrong_mcq_question_ids

def get_mcq_questions(wrong_mcq_question_ids, conn):
    query = """SELECT * FROM mcq_questions WHERE id IN %s"""

    cur = conn.cursor()
    cur.execute(query, (tuple(wrong_mcq_question_ids),))
    rows = cur.fetchall()

    col_names = [desc[0] for desc in cur.description]
    wrong_mcq_questions = [dict(zip(col_names, row)) for row in rows]

    return wrong_mcq_questions


def load_mcq_question_by_id(conn,mcq_question_id):
    cur = conn.cursor()

    query = """
                SELECT *
                FROM mcq_questions
                WHERE id = %s;
            """

    cur.execute(query, (mcq_question_id,))

    row = cur.fetchone()

    col_names = [desc[0] for desc in cur.description]

    result = dict(zip(col_names, row))
    cur.close()
    return result

def insert_custom_quiz(class_id, student_id, question_ids, conn):
    cur = conn.cursor()

    # Convert the list of question IDs to a JSON string
    question_ids_json = json.dumps(question_ids)

    query = """
                INSERT INTO custom_quiz (class_id, user_id, mcq_question_ids)
                VALUES (%s, %s, %s)
                RETURNING id;
            """

    cur.execute(query, (class_id, student_id, question_ids_json))
    row = cur.fetchone()
    conn.commit()
    cur.close()

    return True if row else False