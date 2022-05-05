import axios from 'axios';

const POST_COMMENT_API = 'http://127.0.0.1:5000/sentimentAnalysis';
const POST_SENTIMENT_API = 'http://127.0.0.1:5000/getSentiment';
const GET_SECTION1_QUESTIONS_API = 'http://127.0.0.1:5000/getQuestions/';

class ApiService {
    postComment(Comment){
        return axios.post(POST_COMMENT_API, Comment);
    }
    getSentiment(Comment){
        return axios.post(POST_SENTIMENT_API,Comment);
    }
    getQuestions(section)
    {
        return axios.get(GET_SECTION1_QUESTIONS_API + section);
    }
}

export default new ApiService();
