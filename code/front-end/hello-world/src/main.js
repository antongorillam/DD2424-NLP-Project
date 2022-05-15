import { createApp } from 'vue'
import App from './App.vue'
import PrimeVue from 'primevue/config'

import InputText from "primevue/inputtext";

const app = createApp(App);
app.use(PrimeVue);
app.component("PrimeInputText", InputText);

app.mount('#app')
