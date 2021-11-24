import '../comment.css';
import SubmitButton from './buttonSubmitComment.js';

const inputComment = (props) =>
{
    return (
        <div className="commentBox">
            <h4>INPUT SAMPLE COMMENT</h4>
            <input 
            placeholder="Sample student's comment"
            className="inputComment"
            onChange={props.textChange}
            >
            </input>  
            <SubmitButton 
            submitComment = {props.submitComment}>   
            </SubmitButton>
        </div>
    );
}

export default inputComment