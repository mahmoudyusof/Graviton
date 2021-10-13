import React, {useState} from 'react';
import {TextField, Button, Paper} from "@material-ui/core";
import DropDownMenu from 'material-ui/DropDownMenu';
import MenuItem from 'material-ui/MenuItem';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
import {Typography} from "@material-ui/core";

import { analyze } from '../../api'

import useStyles from './styles'

const Form = () => {

    const tasks = ["Sentiment Analysis", "Question Answering", "Translation", "Text Summarization"]
    const classes = useStyles();

    const [selectedTask, setSelectedTask] = useState({
        selectedTask: tasks[0]
    })
    const [response, setResponse] = useState({
        response: ""
    })


    const handleChange = (event, index, value) => {
        event.preventDefault()
        setSelectedTask({ selectedTask : value });

    }


    const handleSubmit = async (event) => {
        event.preventDefault()

        let text = document.querySelector("#text").value

        let response = await analyze(selectedTask.selectedTask, text)

        setResponse({response: response.data.payload})
    }


    return (
        <>
            <Paper className={classes.paper}>
                <form autoComplete="off" noValidate className={`${classes.root} ${classes.form}`} onSubmit={handleSubmit}>

                    <TextField name="text" varient="outlined" label="Text" fullWidth id="text"/>
                    <MuiThemeProvider>
                        <DropDownMenu value={selectedTask.selectedTask} onChange={handleChange} id="task">
                            {tasks.map((task) => {
                                return <MenuItem value = {task} primaryText={task} />
                            })}

                        </DropDownMenu>
                    </MuiThemeProvider>
                    <Button className={classes.buttonSubmit} variant="contained" color="primary" size="large" type="submit" fullWidth>Analyze</Button>

                </form>
                <br/>
                <Typography>
                    {response.response ? response.response: "Enter text to be analysed and press analyze"}
                </Typography>
            </Paper>
        </>
    );
};

export default Form;