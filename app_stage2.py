import streamlit as st

from classbuild.llm import GenerationError, get_client
from classbuild.models import ChatMessage, Course, CourseSettings, LessonContent, Quiz
from classbuild.rag import (
    COURSE_MATERIAL_KEY,
    DOC_KEY,
    EmbeddingError,
    EmbeddingService,
    ProcessedDocument,
    RetrievalError,
    attach_material_to_course,
    current_results,
    get_or_build_index,
    has_current_index,
    material_for_course,
    run_search,
    sync_learning_material,
)
from classbuild.services import (
    MIN_PASSAGE_SIMILARITY,
    ChatAnswerError,
    ChatInputError,
    GroundedLessonGenerator,
    InsufficientEvidenceError,
    LessonChatContext,
    LessonChatService,
    append_exchange,
    clear_thread,
    generate_course,
    generate_lesson,
    generate_quiz,
    get_history,
    grade_quiz,
    prune_chats,
    readable_citations,
    remove_stale_grounded_lessons,
    thread_key,
    valid_thread_keys,
)

DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]
LESSON_COUNTS = [2, 3, 4, 5]
CHUNK_PREVIEW_COUNT = 5

OUTLINE_KEY = "outline"
SETTINGS_KEY = "course_settings"
COURSE_ID_KEY = "course_id"
LESSONS_KEY = "lessons"  # lesson number -> LessonContent
QUIZZES_KEY = "quizzes"  # lesson number -> Quiz
ATTEMPTS_KEY = "quiz_attempts"  
EVIDENCE_NOTICES_KEY = "evidence_notices"  
CHATS_KEY = "chats"  
UPLOADER_KEY = "pdf_uploader"

# Everything that belongs to one generated course; reset when a new course is generated.
COURSE_STATE_KEYS = {
    OUTLINE_KEY: lambda: None,
    SETTINGS_KEY: lambda: None,
    LESSONS_KEY: dict,
    QUIZZES_KEY: dict,
    ATTEMPTS_KEY: dict,
    COURSE_MATERIAL_KEY: lambda: None,
    EVIDENCE_NOTICES_KEY: dict,
    CHATS_KEY: dict,
}

st.set_page_config(page_title="Mini ClassBuild", layout="wide")


def init_state() -> None:
    """Create every key once; never overwrite existing values."""
    for key, factory in COURSE_STATE_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = factory()
    if COURSE_ID_KEY not in st.session_state:
        st.session_state[COURSE_ID_KEY] = 0


def reset_course_state() -> None:
    for key, factory in COURSE_STATE_KEYS.items():
        st.session_state[key] = factory()
    st.session_state.pop("lesson_picker", None)


def current_pdf_fingerprint() -> str | None:
    document = st.session_state.get(DOC_KEY)
    return document.fingerprint if document is not None else None


def fingerprint_tag(document: ProcessedDocument | None) -> str:
    return document.fingerprint[:12] if document is not None else "nopdf"


def render_learning_materials() -> None:
    st.header("Learning material (optional)")
    uploaded = st.file_uploader(
        "Upload a PDF to use as reference (maximum 10 MB)", type=["pdf"], key=UPLOADER_KEY
    )
    upload = (uploaded.name, uploaded.getvalue()) if uploaded is not None else None
    status = sync_learning_material(st.session_state, upload)

    if status.error:
        st.error(status.error)
    elif status.document is not None:
        render_document_summary(status.document)
        render_pdf_search(status.document)
    else:
        st.caption("No PDF loaded. Lessons and chat work without one.")


def render_document_summary(document: ProcessedDocument) -> None:
    st.success(
        f"Extracted text from {document.text_page_count} of {document.page_count} pages "
        f"of {document.source_filename}."
    )
    pages, words, chunks = st.columns(3)
    pages.metric("Pages", document.page_count)
    words.metric("Words", document.word_count)
    chunks.metric("Chunks", document.chunk_count)
    if document.empty_pages:
        st.warning("No text found on page(s): " + ", ".join(str(p) for p in document.empty_pages))
    with st.expander("Chunk preview"):
        for chunk in document.chunks[:CHUNK_PREVIEW_COUNT]:
            st.caption(f"{chunk.chunk_id} | page {chunk.page_number} | {chunk.char_count} characters")
            st.text(chunk.text)  # PDF text is untrusted: never rendered as Markdown
        if document.chunk_count > CHUNK_PREVIEW_COUNT:
            st.caption(f"Showing {CHUNK_PREVIEW_COUNT} of {document.chunk_count} chunks.")


def render_pdf_search(document: ProcessedDocument) -> None:
    tag = fingerprint_tag(document)
    st.subheader("Search the PDF")
    with st.form(f"pdf_search_form_{tag}"):
        query = st.text_input("Question or topic", key=f"pdf_search_query_{tag}")
        submitted = st.form_submit_button("Search PDF")
    if submitted:
        try:
            service = EmbeddingService()
            if not has_current_index(st.session_state, document, service):
                with st.spinner("Building the search index (the first run downloads the embedding model)..."):
                    get_or_build_index(st.session_state, document, service)
            run_search(st.session_state, query, service)
        except (RetrievalError, EmbeddingError) as exc:
            st.error(str(exc))
    outcome = current_results(st.session_state)
    if outcome is not None:
        render_search_results(outcome.query, outcome.results)


def render_search_results(query: str, results) -> None:
    st.markdown(f"**Top results for:** {query}")
    st.caption(
        "Similarity shows how close a passage is in meaning. It does not prove that an answer is correct."
    )
    for result in results:
        weak = result.score < MIN_PASSAGE_SIMILARITY
        label = f"#{result.rank} | page {result.page_number} | {result.chunk_id} | score {result.score:.3f}"
        st.markdown(f"**{label}**" + ("  \nLow similarity: not treated as evidence." if weak else ""))
        st.text(result.text)


def render_course_material_note() -> None:
    document = st.session_state.get(DOC_KEY)
    if document is None:
        return
    if material_for_course(st.session_state) is not None:
        st.caption(f"This course was generated with {document.source_filename} loaded.")
    else:
        st.caption(
            f"{document.source_filename} was loaded after this course was generated. "
            "It can still be used as a reference for lessons."
        )


def render_form() -> None:
    st.header("Course")
    col_topic, col_difficulty, col_count = st.columns([3, 1, 1])
    with col_topic:
        topic = st.text_input("Course Topic", key="topic_input", placeholder="For example: Introduction to Python")
    with col_difficulty:
        difficulty = st.selectbox("Difficulty", DIFFICULTIES, key="difficulty_input")
    with col_count:
        count = st.selectbox("Number of lessons", LESSON_COUNTS, key="count_input")

    if not st.button("Generate Course Outline", type="primary"):
        return
    topic = topic.strip()
    if not topic:
        st.warning("Please enter a course topic before generating an outline.")
        return
    settings = CourseSettings(topic=topic, difficulty=difficulty, lesson_count=count)
    try:
        with st.spinner("Generating course outline..."):
            course = generate_course(get_client(), settings)
    except GenerationError as exc:
        st.error(str(exc))  # an earlier outline stays on screen
        return

    reset_course_state()  # a new course starts clean
    st.session_state[COURSE_ID_KEY] += 1
    st.session_state[OUTLINE_KEY] = course
    st.session_state[SETTINGS_KEY] = settings
    attach_material_to_course(st.session_state)  # remember which PDF was loaded


def render_outline() -> None:
    outline: Course | None = st.session_state[OUTLINE_KEY]
    if outline is None:
        st.info("No course yet. Enter a topic above and select Generate Course Outline.")
        return
    settings: CourseSettings = st.session_state[SETTINGS_KEY]

    drop_stale_grounded_lessons()
    prune_stale_chats()

    st.caption(f"{settings.topic} | {settings.difficulty} | {settings.lesson_count} lessons")
    st.subheader(outline.title)
    st.write(outline.description)
    st.markdown("**Course objectives**")
    for objective in outline.objectives:
        st.markdown(f"- {objective}")
    st.markdown("**Lessons**")
    for number, lesson in enumerate(outline.lessons, start=1):
        with st.expander(f"Lesson {number}: {lesson.title}"):
            st.write(lesson.summary)
            for objective in lesson.objectives:
                st.markdown(f"- {objective}")

    st.divider()
    st.header("Lesson")
    render_course_material_note()
    document = render_pdf_reference_option()
    render_lesson_section(outline, settings, document)


def render_pdf_reference_option() -> ProcessedDocument | None:
    """Checkbox (off by default). Returns the document to ground new lessons in, or None."""
    document = st.session_state.get(DOC_KEY)
    if document is None:
        return None
    key = f"use_pdf_{st.session_state[COURSE_ID_KEY]}_{fingerprint_tag(document)}"
    enabled = st.checkbox(
        "Use Uploaded PDF as Reference",
        key=key,
        value=False,
        help="Ground newly generated lessons in passages from the PDF and show page citations.",
    )
    return document if enabled else None


def generate_grounded_lesson(outline: Course, settings: CourseSettings, lesson_number: int, document) -> LessonContent:
    service = EmbeddingService()
    index = get_or_build_index(st.session_state, document, service)  # built once per PDF
    generator = GroundedLessonGenerator(get_client(), service)
    return generator.generate_lesson(outline, settings, lesson_number, document, index)


def generate_lesson_content(outline: Course, settings: CourseSettings, lesson_number: int, document=None) -> None:
    notices = st.session_state[EVIDENCE_NOTICES_KEY]
    try:
        with st.spinner("Generating lesson..."):
            if document is not None:
                lesson = generate_grounded_lesson(outline, settings, lesson_number, document)
            else:
                lesson = generate_lesson(get_client(), outline, settings, lesson_number)
    except InsufficientEvidenceError as exc:
        notices[lesson_number] = {"message": str(exc), "fingerprint": document.fingerprint}
        return
    except (RetrievalError, EmbeddingError) as exc:
        st.error(str(exc))
        return
    except GenerationError as exc:
        st.error(str(exc))
        return

    notices.pop(lesson_number, None)
    st.session_state[LESSONS_KEY][lesson_number] = lesson
    # A regenerated lesson invalidates its old quiz and answers.
    st.session_state[QUIZZES_KEY].pop(lesson_number, None)
    st.session_state[ATTEMPTS_KEY].pop(lesson_number, None)


def current_evidence_notice(lesson_number: int) -> str | None:
    """A lesson's 'not enough evidence' notice, only if it refers to the loaded PDF."""
    notice = st.session_state[EVIDENCE_NOTICES_KEY].get(lesson_number)
    if notice is not None and notice["fingerprint"] == current_pdf_fingerprint():
        return notice["message"]
    return None


def drop_stale_grounded_lessons() -> None:
    removed = remove_stale_grounded_lessons(
        st.session_state[LESSONS_KEY],
        current_pdf_fingerprint(),
        st.session_state[QUIZZES_KEY],
        st.session_state[ATTEMPTS_KEY],
    )
    if removed:
        names = ", ".join(str(n) for n in removed)
        st.warning(f"Lesson(s) {names} were based on a PDF that is no longer loaded and were removed.")


def prune_stale_chats() -> None:
    valid = valid_thread_keys(
        st.session_state[COURSE_ID_KEY], st.session_state[LESSONS_KEY], current_pdf_fingerprint()
    )
    prune_chats(st.session_state[CHATS_KEY], valid)


def render_lesson_content(content: LessonContent) -> None:
    if content.is_grounded:
        pages = ", ".join(str(page) for page in content.cited_pages)
        st.caption(f"Based on {content.grounding.source_filename} | pages {pages}")
    st.markdown("#### Introduction")
    st.write(content.introduction)
    pages_by_id = {s.chunk_id: s.page_number for s in content.grounding.sources} if content.grounding else {}
    for number, section in enumerate(content.sections, start=1):
        st.markdown(f"#### {number}. {section.heading}")
        st.write(section.explanation)
        st.markdown("**Example**")
        st.write(section.example)
        if section.source_ids:
            st.caption("Sources: " + ", ".join(f"{i} (page {pages_by_id[i]})" for i in section.source_ids))
    st.markdown("#### Summary")
    st.write(content.summary)
    st.markdown("#### Exercises")
    for number, exercise in enumerate(content.exercises, start=1):
        st.markdown(f"**Exercise {number}.** {exercise.task}")
        with st.expander("Show hint"):
            st.write(exercise.hint)
    if content.is_grounded:
        render_lesson_sources(content)


def render_lesson_sources(content: LessonContent) -> None:
    st.markdown("#### Sources")
    for source in content.grounding.sources:
        st.markdown(
            f"**{source.chunk_id}** | {source.source_filename} | page {source.page_number} | "
            f"similarity {source.similarity:.2f}"
        )
        with st.expander("Show passage"):
            st.text(source.text)  # PDF text is untrusted: never rendered as Markdown


def render_lesson_section(outline: Course, settings: CourseSettings, document) -> None:
    course_id = st.session_state[COURSE_ID_KEY]
    index = st.selectbox(
        "Select a lesson",
        options=list(range(len(outline.lessons))),
        format_func=lambda i: f"Lesson {i + 1}: {outline.lessons[i].title}",
        key="lesson_picker",
    )
    number = index + 1
    lessons = st.session_state[LESSONS_KEY]
    content: LessonContent | None = lessons.get(number)

    notice = current_evidence_notice(number)
    if notice is not None:
        st.warning(f"{notice} The lesson was not generated from the PDF.")
        if st.button("Generate Without PDF", key=f"nopdf_{course_id}_{number}"):
            generate_lesson_content(outline, settings, number, None)
            st.rerun()

    if content is None:
        if st.button("Generate Lesson Content", key=f"gen_lesson_{course_id}_{number}"):
            generate_lesson_content(outline, settings, number, document)
            content = lessons.get(number)
            if content is None and current_evidence_notice(number) is not None:
                st.rerun()  # show the notice with its Generate Without PDF button
        elif notice is None:
            st.info("This lesson has not been generated yet.")
    else:
        if st.button("Regenerate Lesson", key=f"regen_{course_id}_{number}"):
            generate_lesson_content(outline, settings, number, document)
            st.rerun()

    if content is not None:
        render_lesson_content(content)
        st.divider()
        render_quiz_section(outline, settings, number, content, course_id)
        st.divider()
        render_lesson_chat(outline, settings, content)

def render_quiz_section(outline, settings, number: int, content: LessonContent, course_id: int) -> None:
    st.markdown("### Quiz")
    quizzes = st.session_state[QUIZZES_KEY]
    quiz: Quiz | None = quizzes.get(number)
    if quiz is None:
        if st.button("Generate Quiz", key=f"gen_quiz_{course_id}_{number}"):
            try:
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(get_client(), outline, settings, number, content)
                quizzes[number] = quiz
            except GenerationError as exc:
                st.error(str(exc))
        else:
            st.info("Generate a five-question quiz for this lesson.")
    if quiz is not None:
        render_quiz(quiz, number, course_id)


def render_quiz(quiz: Quiz, number: int, course_id: int) -> None:
    attempt = st.session_state[ATTEMPTS_KEY].setdefault(number, {"attempt": 0, "result": None})

    if attempt["result"] is None:
        form_id = f"quiz_form_{course_id}_{number}_{attempt['attempt']}"
        with st.form(form_id):
            choices = []
            for position, question in enumerate(quiz.questions, start=1):
                choices.append(
                    st.radio(
                        f"{position}. {question.question}",
                        options=list(range(len(question.options))),
                        format_func=lambda i, q=question: q.options[i],
                        index=None,
                        key=f"{form_id}_q{position}",
                    )
                )
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            if any(choice is None for choice in choices):
                st.warning("Please answer all 5 questions before submitting.")
            else:
                attempt["result"] = grade_quiz(quiz, choices)  # graded locally
                st.rerun()
        return

    result = attempt["result"]
    st.metric("Score", f"{result['score']} / {len(quiz.questions)}")
    for position, question in enumerate(quiz.questions, start=1):
        chosen = result["answers"][position - 1]
        st.markdown(f"**{position}. {question.question}**")
        st.write(f"Your answer: {question.options[chosen]}")
        if result["correct"][position - 1]:
            st.success("Correct.")
        else:
            st.error(f"Incorrect. Correct answer: {question.options[question.correct_index]}")
        st.caption(question.explanation)
    if st.button("Retake quiz", key=f"retake_{course_id}_{number}"):
        attempt["attempt"] += 1  # new widget keys clear the old answers
        attempt["result"] = None
        st.rerun()


def render_lesson_chat(outline: Course, settings: CourseSettings, content: LessonContent) -> None:
    course_id = st.session_state[COURSE_ID_KEY]
    document = st.session_state.get(DOC_KEY)
    st.markdown("### Ask about this lesson")

    use_pdf = False
    if document is not None:
        use_pdf = st.checkbox(
            "Answer from the uploaded PDF",
            key=f"chat_pdf_{course_id}_{content.lesson_number}_{fingerprint_tag(document)}",
            value=False,
        )
    active_document = document if use_pdf else None
    key = thread_key(course_id, content, active_document.fingerprint if active_document else None)
    chats = st.session_state[CHATS_KEY]

    question = st.chat_input(
        "Ask a question about this lesson", key=f"chat_input_{course_id}_{content.lesson_number}"
    )
    if question is not None:
        answer_question(outline, settings, content, question, get_history(chats, key), key, active_document)

    history = get_history(chats, key)
    for message in history:
        render_chat_message(message)
    if history and st.button("Clear Chat", key=f"clear_chat_{course_id}_{content.lesson_number}"):
        clear_thread(chats, key)
        st.rerun()


def answer_question(outline, settings, content, question, history, key, document) -> None:
    """Ask the tutor; the exchange is saved only after a complete answer."""
    try:
        with st.spinner("Thinking..."):
            service = LessonChatService(get_client(), EmbeddingService())
            index = get_or_build_index(st.session_state, document, service.embedding_service) if document else None
            ctx = LessonChatContext.from_course(outline, settings, content)
            reply = service.answer(ctx, question, history, document, index)
    except ChatInputError as exc:
        st.warning(str(exc))
    except (ChatAnswerError, GenerationError, RetrievalError, EmbeddingError) as exc:
        st.error(str(exc))
    else:
        append_exchange(st.session_state[CHATS_KEY], key, reply.question, reply.answer)


def render_chat_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        render_answer_body(message)


def render_answer_body(message: ChatMessage) -> None:
    if message.role == "assistant" and message.mode == "pdf" and message.sources:
        st.write(readable_citations(message.content, message.sources))
        pages = ", ".join(f"p. {page}" for page in message.cited_pages)
        st.caption(f"Sources: {message.sources[0].source_filename} ({pages})")
        with st.expander("Show passages"):
            for source in message.sources:
                st.caption(f"{source.chunk_id} | page {source.page_number} | similarity {source.similarity:.2f}")
                st.text(source.text)  # PDF text is untrusted
    else:
        st.write(message.content)
        if message.truncated:
            st.caption("This answer was cut off because it reached the length limit.")


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Mini ClassBuild")
        st.write(
            "Generate a course outline, full lessons and quizzes. Optionally upload a PDF to search it "
            "by meaning, ground lessons in it with page citations, and ask a tutor about each lesson."
        )


def main() -> None:
    init_state()
    render_sidebar()
    st.title("Mini ClassBuild")
    st.write(
        "Enter a topic, choose a difficulty and a lesson count. Mini ClassBuild generates a course "
        "outline, full lessons and a five-question quiz for each lesson."
    )
    render_learning_materials()
    st.divider()
    render_form()
    st.divider()
    render_outline()


main()
