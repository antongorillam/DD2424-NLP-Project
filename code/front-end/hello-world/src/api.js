import axios from "axios";

const BASE = "https://dd2482-22.azurewebsites.net/"

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
