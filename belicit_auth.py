import streamlit as st
from supabase import create_client

@st.cache_resource
def _sb():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])

def require_login():
    if "user" in st.session_state:
        return st.session_state["user"]
    code = st.query_params.get("code")
    if code:
        st.query_params.clear()
        res = _sb().rpc("redeem_sso_code", {"p_code": code}).execute()
        if res.data:
            st.session_state["user"] = res.data
            st.rerun()
        st.error("Login expired. Please try again.")
    st.markdown("### Log in to use WAL.AI")
    st.link_button("Log in with Belicit", "https://www.belicit.com/wal-login.html")
    st.stop()
