<template>
  <div class="hello">
    <div class="settings">
      <h3 class="settings-text"> Settings: </h3>
      <p class="words-text"> Number of words: </p>
      <input class="words-input" v-model="num_of_words"/>
      <p class="initial-text"> Initial word: </p>
      <input class="initial-input" v-model="initial_string"/>
    </div>
    <input-text type="text" v-model="value" placeholder=msg />
    <button class="ml-button" @click="getText"> Click to Run</button>
    <center>
      <div v-if="isLoading" class="loader"></div>
    </center>
    <p v-if="!isLoading" class="output">{{value}}</p>
  </div>
</template>

<script>
import Apis from "../api"

export default {
  name: 'HelloWorld',
  data() {
    return {
      value: "Click to use ML model to generate a shakespear text...",
      num_of_words: 100,
      initial_string: "a",
      isLoading: false,
    }
  },
  props: {
    msg: String
  },
  methods: {
    async getText() {
      this.isLoading = true;
      //const val = await Apis.getSynthesizedText(this.num_of_words, this.initial_string)
      //console.log(val)
      this.value = await Apis.getSynthesizedText(this.num_of_words, this.initial_string)
      this.value = this.value.message
      this.isLoading = false;
    }
  }
}
</script>

<style scoped>
.hello{
  width:100%;
  color: black;
}

.ml-button {
  margin-top: 2%;
  width: 150px;
  height: 30px;
  background-color: #008CBA;
  color: white;
  border: none;
  border-radius: 4px;
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
}

.settings {
  display: grid;
  grid-template-columns: 33% 33% 33%;
  grid-template-areas: 
  "... settings-text ..."
  "words-text words-input ..."
  "initial-text initial-input ...";
}

.settings-text {
  grid-area: settings-text;
}

.words-text {
  grid-area: words-text;
}

.words-input {
  grid-area: words-input;
  height: 20px;
  margin-top: 15px;
}

.initial-text {
  grid-area: initial-text;
}

.initial-input {
  height: 20px;
  grid-area: initial-input;
  margin-top: 15px;
}

.output {
  text-align: justify;
  text-justify: inter-word;
  margin-left: 10%;
  margin-right: 10%;
  text-align: center;
  margin-top: 5%;
}

.loader {
  place-items: center;
  border: 16px solid #f3f3f3; /* Light grey */
  border-top: 16px solid #3498db; /* Blue */
  border-radius: 50%;
  width: 80px;
  height: 80px;
  margin-top: 5%;
  animation: spin 2s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
