import streamlit as st
from main import graph


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Report AI",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🩺 Medical Report AI Assistant")

st.write(
    "Analyze a laboratory report using a multi-agent AI pipeline "
    "with RAG-based medical knowledge retrieval."
)

st.warning(
    "⚠️ Educational prototype only. This system does not diagnose "
    "medical conditions or prescribe treatment."
)


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("📄 Enter Medical Report")

uploaded_file = st.file_uploader(
    "Upload a text report",
    type=["txt"]
)

report_text = ""

if uploaded_file is not None:

    report_text = uploaded_file.read().decode("utf-8")

else:

    report_text = st.text_area(
        "Paste your laboratory report here",
        height=250,
        placeholder="""Example:

PATIENT LABORATORY REPORT

Test Name: HbA1c
Result: 8.2 %
Reference Range: 4.0 - 5.6 %

Test Name: Hemoglobin
Result: 13.5 g/dL
Reference Range: 13.0 - 17.0 g/dL
"""
    )


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("🔍 Analyze Report", use_container_width=True):

    if report_text.strip() == "":

        st.error("Please enter or upload a medical report.")

    else:

        with st.spinner("Running medical report analysis..."):

            result = graph.invoke({
                "report_text": report_text
            })

        st.success("Analysis completed successfully.")


        # --------------------------------------------------
        # AGENT 5 - FINAL RESPONSE
        # --------------------------------------------------

        final_response = result["agent5response"]

        st.divider()

        st.subheader("📋 Final Report")

        st.info(final_response["summary"])


        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------

        st.markdown("### 🧠 Explanation")

        st.write(
            final_response["explanation"]
        )


        # --------------------------------------------------
        # PRIORITY
        # --------------------------------------------------

        st.markdown("### ⚠️ Attention Priority")

        priority = final_response["priority"]

        if priority == "HIGH":

            st.error(
                f"Priority: {priority}"
            )

        elif priority == "MEDIUM":

            st.warning(
                f"Priority: {priority}"
            )

        else:

            st.success(
                f"Priority: {priority}"
            )


        # --------------------------------------------------
        # NEXT STEPS
        # --------------------------------------------------

        st.markdown("### ➡️ Suggested Next Steps")

        st.write(
            final_response["next_steps"]
        )


        # --------------------------------------------------
        # MISSING INFORMATION
        # --------------------------------------------------

        st.markdown("### ℹ️ Missing Information")

        missing_info = final_response["missing_info"]

        if len(missing_info) == 0:

            st.write("No missing information identified.")

        else:

            for item in missing_info:

                st.write(f"- {item}")


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.markdown("### ⚕️ Disclaimer")

        st.caption(
            final_response["disclaimer"]
        )


        # ==================================================
        # AGENT PIPELINE
        # ==================================================

        st.divider()

        st.subheader("🔎 Agent Pipeline")


        # --------------------------------------------------
        # AGENT 1
        # --------------------------------------------------

        with st.expander("Agent 1 — 📄 Report Extraction"):

            agent1 = result["agent1response"]

            st.write(
                "**Test Name:**",
                agent1["test_name"]
            )

            st.write(
                "**Value:**",
                agent1["value"]
            )

            st.write(
                "**Unit:**",
                agent1["unit"]
            )

            st.write(
                "**Reference Range:**",
                agent1["reference_range"]
            )

            st.markdown("**Extracted Report Data:**")

            st.write(
                agent1["report_data"]
            )


        # --------------------------------------------------
        # AGENT 2
        # --------------------------------------------------

        with st.expander("Agent 2 — 🔎 RAG Interpretation"):

            agent2 = result["agent2response"]

            st.write(
                "**Test Name:**",
                agent2["test_name"]
            )

            st.write(
                "**Value:**",
                agent2["value"],
                agent2["unit"]
            )

            st.write(
                "**Reference Range:**",
                agent2["reference_range"]
            )

            st.markdown("**Interpretation:**")

            st.write(
                agent2["interpretation"]
            )


        # --------------------------------------------------
        # AGENT 3
        # --------------------------------------------------

        with st.expander("Agent 3 — ⚠️ Priority Analysis"):

            agent3 = result["agent3response"]

            st.write(
                "**Test Name:**",
                agent3["test_name"]
            )

            st.write(
                "**Value:**",
                agent3["value"],
                agent3["unit"]
            )

            st.write(
                "**Reference Range:**",
                agent3["reference_range"]
            )

            st.markdown("**Interpretation:**")

            st.write(
                agent3["interpretation"]
            )

            st.markdown("**Attention Priority:**")

            if agent3["priority"] == "HIGH":

                st.error(
                    agent3["priority"]
                )

            elif agent3["priority"] == "MEDIUM":

                st.warning(
                    agent3["priority"]
                )

            else:

                st.success(
                    agent3["priority"]
                )

            st.markdown("**Reason:**")

            st.write(
                agent3["reason"]
            )


        # --------------------------------------------------
        # AGENT 4
        # --------------------------------------------------

        with st.expander("Agent 4 — ✅ Validation"):

            agent4 = result["agent4response"]

            if agent4["valid"]:

                st.success("Validation: VALID")

            else:

                st.error("Validation: INVALID")


            st.markdown("**Missing Fields:**")

            if len(agent4["missing_fields"]) == 0:

                st.write("None")

            else:

                for field in agent4["missing_fields"]:

                    st.write(
                        f"- {field}"
                    )


            st.markdown("**Validation Reason:**")

            st.write(
                agent4["reason"]
            )


        # --------------------------------------------------
        # AGENT 5 DETAILS
        # --------------------------------------------------

        with st.expander("Agent 5 — 🧠 Final Response Generation"):

            st.write(
                "**Summary:**"
            )

            st.write(
                final_response["summary"]
            )

            st.write(
                "**Explanation:**"
            )

            st.write(
                final_response["explanation"]
            )

            st.write(
                "**Priority:**",
                final_response["priority"]
            )

            st.write(
                "**Next Steps:**"
            )

            st.write(
                final_response["next_steps"]
            )

            st.write(
                "**Missing Information:**"
            )

            if len(final_response["missing_info"]) == 0:

                st.write("None")

            else:

                for item in final_response["missing_info"]:

                    st.write(
                        f"- {item}"
                    )