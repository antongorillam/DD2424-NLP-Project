<template>
  <div class="hello">
    lolp
    <input-text type="text" v-model="value" placeholder=msg />
    <button @click="getText"/>
    {{value}}
  </div>
</template>

<script>
import Apis from "../api"

export default {
  name: 'HelloWorld',
  data() {
    return {
      value: "lolp",
      num_of_words: 100,
      initial_string: "a"
    }
  },
  props: {
    msg: String
  },
  methods: {
    async getText() {
      //const val = await Apis.getSynthesizedText(this.num_of_words, this.initial_string)
      //console.log(val)
      this.value = await Apis.getSynthesizedText(this.num_of_words, this.initial_string)
      this.value = this.value.message
    }
  }
}
</script>

<style scoped>
.hello{
  width:100%;
  color: black;
}
</style>
