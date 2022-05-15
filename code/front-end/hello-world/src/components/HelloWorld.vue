<template>
  <div class="hello">
    lolp
    <input-text type="text" v-model="value" placeholder=msg />
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>

<style scoped>
.hello{
  width:100%;
  color: black;
}
</style>
