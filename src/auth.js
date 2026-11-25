import { createClient } from '@supabase/supabase-js';

const sb = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY);
const KEY = 'wal_user';
export const LOGIN_URL = 'https://www.belicit.com/wal-login.html';

export const getUser = () => {
  try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; }
};
export const logout = () => localStorage.removeItem(KEY);

async function redeem() {
  const code = new URLSearchParams(location.search).get('code');
  if (!code) return { user: getUser() };
  history.replaceState({}, '', location.pathname);
  const { data, error } = await sb.rpc('redeem_sso_code', { p_code: code });
  if (error || !data) return { user: null, error: 'Login expired. Please try again.' };
  let user = Array.isArray(data) ? data[0] : data;
  if (typeof user !== 'object' || user === null) user = { id: String(user) };

  // Best effort: if the SSO result carries no username, look it up in the shared "profiles" table.
  if (!describeUser(user).username && (user.id || user.user_id)) {
    try {
      const id = user.id || user.user_id;
      const { data: profile } = await sb.from('profiles').select('*').eq('id', id).maybeSingle();
      if (profile) user = { ...user, profile };
    } catch { /* optional enrichment only */ }
  }
  localStorage.setItem(KEY, JSON.stringify(user));
  return { user };
}

let pending;
export const redeemFromUrl = () => (pending ??= redeem());

const pick = (o, keys) => keys.map((k) => o[k]).find((v) => typeof v === 'string' && v.trim());

export function describeUser(raw) {
  const u = Array.isArray(raw) ? raw[0] : raw;
  const src = { ...(u?.user || {}), ...(u?.profile || {}), ...(u || {}) };
  const username = pick(src, ['username', 'user_name', 'handle', 'nickname']);
  const name = pick(src, ['display_name', 'full_name', 'name']);
  const email = pick(src, ['email']);
  return {
    id: String(src.id || src.user_id || ''),
    username, name, email,
    created: src.created_at || src.joined_at || '',
    display: username || name || (email ? email.split('@')[0] : 'there'),
  };
}
