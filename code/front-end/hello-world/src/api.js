import axios from "axios";

const BASE = "http://127.0.0.1:8081/"

export function getSynthesizedText(num_words, initial_word){
  return new Promise((resolve, reject) => {
    axios.get(BASE+"Synthesize", {
      params: {num_words, initial_word}
    }).then((response) => {
      console.log("went here")
      console.log(response.data)
      resolve(response.data)
    }).catch((error)=> {
      reject(error)
    })
  })
}

export default{
  getSynthesizedText
}
