/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                darkBg: '#0b0f19',
                darkPanel: '#111827',
                primary: '#3b82f6',
                accent: '#10b981',
                danger: '#ef4444'
            }
        },
    },
    plugins: [],
}
