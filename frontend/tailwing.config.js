/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#0b0f17',
        panel: '#0f1523',
      },
      boxShadow: { soft: '0 8px 30px rgba(0,0,0,0.25)' },
      borderRadius: { '2xl': '1rem' },
      transitionTimingFunction: { smooth: 'cubic-bezier(0.22, 1, 0.36, 1)' },
    },
  },
  plugins: [],
}
